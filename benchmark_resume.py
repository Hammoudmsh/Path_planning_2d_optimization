"""Durable per-search journal and reuse of measured A* references."""
import json
import sqlite3
from pathlib import Path


class SearchJournal:
    def __init__(self, path, signature, reset=False):
        self.connection = sqlite3.connect(path)
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (signature TEXT)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS runs '
                                '(map TEXT, scenario INTEGER, algorithm TEXT, payload TEXT, '
                                'PRIMARY KEY(map, scenario, algorithm))')
        identity = json.dumps(signature, sort_keys=True, default=str)
        previous = self.connection.execute('SELECT signature FROM metadata').fetchone()
        if previous and previous[0] != identity and not reset:
            self.connection.close()
            raise ValueError('Journal signature mismatch; refusing to reuse unrelated results')
        with self.connection:
            if reset:
                self.connection.execute('DELETE FROM runs')
                self.connection.execute('DELETE FROM metadata')
            if reset or not previous:
                self.connection.execute('INSERT INTO metadata VALUES (?)', (identity,))

    def rows(self):
        return [json.loads(row[0]) for row in self.connection.execute(
            'SELECT payload FROM runs ORDER BY map, scenario, algorithm')]

    def save(self, row):
        with self.connection:
            self.connection.execute('INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?)', (
                row['Map name'], int(row['Source scenario index']), row['Algorithm'],
                json.dumps(row, default=lambda value: value.item())))

    def close(self):
        self.connection.close()


def reuse_astar_references(frame, config):
    """Join by map, original scenario index AND endpoints; never substitute .scen lengths."""
    import numpy as np
    import pandas as pd

    root = config.get('astar_reference_root')
    if not root:
        return frame
    dataset = config['dataset_name']
    source = Path(root) / dataset / 'full' / f'per_run_results_{dataset}.csv'
    if not source.exists():
        print(f'A* reference file missing: {source}; A* gaps remain unavailable.')
        return frame
    keys = ['Map name', 'Source scenario index', 'Start x', 'Start y', 'Goal x', 'Goal y']
    reference = pd.read_csv(source)
    if not set(keys + ['A* reference length']).issubset(reference.columns):
        raise ValueError(f'Reference file lacks scenario identity or A* lengths: {source}')
    reference = reference[keys + ['A* reference length']].dropna().drop_duplicates()
    if reference.duplicated(keys).any():
        raise ValueError(f'Conflicting A* reference lengths in {source}')
    result = frame.drop(columns=['A* reference length', 'Gap vs A* percent'], errors='ignore')
    result = result.merge(reference, on=keys, how='left', validate='many_to_one')
    denominator = result['A* reference length'].replace(0, np.nan)
    result['Gap vs A* percent'] = 100 * (result['Path length'] - denominator) / denominator
    zero = (result['A* reference length'] == 0) & (result['Path length'] == 0)
    result.loc[zero, 'Gap vs A* percent'] = 0.0
    result.loc[~result['Success'].astype(bool), 'Gap vs A* percent'] = np.nan
    print(f'Reused A* references: {result["A* reference length"].notna().sum()}/{len(result)} from {source}')
    return result
