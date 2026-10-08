"""Small interruption test; no dataset download or full benchmark execution."""
import ast
import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_resume import SearchJournal, reuse_astar_references


class Progress:
    def __init__(self, **kwargs): pass
    def update(self, *args): pass
    def set_description(self, *args): pass
    def set_postfix(self, *args, **kwargs): pass
    def close(self): pass


class ResumeTests(unittest.TestCase):
    def test_default_experiment_dispatches_only_weighted_jps(self):
        from types import SimpleNamespace
        source = Path(__file__).with_name('Astar_JPS_SA_JPS_config_profiles_ready.py')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'run_all_algorithms')
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
            ast.alias(name='annotations')], level=0), function], type_ignores=[])
        calls = []
        def search(*args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(success=False)
        namespace = {'ALGORITHM_ORDER': ['Standard JPS'], 'run_jps': search}
        exec(compile(ast.fix_missing_locations(module), '<dispatcher>', 'exec'), namespace)
        config = json.loads(source.with_name('exp_config.json').read_text())['overrides']
        result = namespace['run_all_algorithms'](None, (0, 0), (1, 1),
                                                config['benchmark_approaches'])
        self.assertEqual(calls, [{'algorithm_name': 'Weighted JPS',
                                 'heuristic_weight': 1.15, 'smooth_tie_break': False}])
        self.assertEqual(result[0].turn_weight, 0.0)
        self.assertEqual(result[0].heuristic_weight, 1.15)
        self.assertFalse(config['run_ablation_study'])

    def test_interrupted_benchmark_skips_committed_searches(self):
        # Load the real loop without executing notebook-style module globals.
        tree = ast.parse(Path(__file__).with_name(
            'Astar_JPS_SA_JPS_config_profiles_ready.py').read_text(encoding='utf-8'))
        wanted = {'benchmark_dataset', '_completed_execution_keys'}
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
            ast.alias(name='annotations')], level=0)] + [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenarios = pd.DataFrame([dict(start_x=i, start_y=0, goal_x=3,
                goal_y=0, reference_length=3-i, source_scenario_index=10+i)
                for i in range(3)])
            features = {key: 0.0 for key in ['Obstacle density', 'Free-cell ratio',
                'Dead-end ratio', 'Corridor ratio', 'Junction ratio',
                'Obstacle-edge density', 'Mean axis visibility',
                'Connected components', 'Largest component ratio']}
            calls = []
            interrupt = [True]

            def search(grid, start, goal, approaches):
                label = approaches[0]['label']
                calls.append((start[0], label))
                if start[0] == 0 and label == 'Fast Weighted SA-JPS' and interrupt[0]:
                    raise KeyboardInterrupt('simulated interruption')
                class Result:
                    success = True
                    def metrics(self):
                        return {'Algorithm': label, 'Path length': 3-start[0],
                                'Success': True, 'Time ms': 1.0}
                return [Result()]

            namespace = dict(pd=pd, np=np, Path=Path, time=time, SearchJournal=SearchJournal,
                CONFIG={'scenario_selection': 'uniform', 'scenarios_per_map': 200},
                ALGORITHM_ORDER=['Weighted JPS', 'Fast Weighted SA-JPS'], RESULTS_DIR=root,
                SCENARIOS_DIR=root, selected_maps=[root/'tiny.map'], tqdm=Progress,
                normalise_approaches=lambda a: a,
                _benchmark_cache_signature=lambda a: {'approaches': a},
                _benchmark_cache_paths=lambda s: (root/'cache.pkl', root/'cache.json'),
                _benchmark_checkpoint_paths=lambda p: (root/'old.pkl', root/'old.json'),
                _load_benchmark_checkpoint=lambda *a: None,
                read_map=lambda p: np.ones((4, 4)),
                get_map_type=lambda g: ('Open', features),
                locate_file=lambda *a: root/'tiny.scen', read_scenario=lambda p: scenarios,
                select_scenarios=lambda *a, **k: scenarios,
                obstacle_percentage=lambda g: 0, run_all_algorithms=search)
            exec(compile(ast.fix_missing_locations(module), '<benchmark-loop>', 'exec'), namespace)
            run = namespace['benchmark_dataset']
            approaches = [{'label': 'Weighted JPS', 'algorithm': 'Standard JPS',
                           'parameters': {'heuristic_weight': 1.15}},
                          {'label': 'Fast Weighted SA-JPS', 'algorithm': 'Fast Weighted SA-JPS',
                           'parameters': {'heuristic_weight': 1.15, 'turn_weight': 0.35}}]
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    run(approaches=approaches, use_cache=False)
                interrupt[0] = False
                frame, _ = run(approaches=approaches, use_cache=False)
                again, _ = run(approaches=approaches, use_cache=False)
            self.assertEqual(calls, [(0, 'Weighted JPS'), (0, 'Fast Weighted SA-JPS'),
                                    (0, 'Fast Weighted SA-JPS'), (1, 'Weighted JPS'),
                                    (1, 'Fast Weighted SA-JPS'), (2, 'Weighted JPS'),
                                    (2, 'Fast Weighted SA-JPS')])
            self.assertEqual(len(frame), 6)
            self.assertEqual(len(again), 6)
            self.assertFalse(frame.duplicated(['Map name', 'Source scenario index', 'Algorithm']).any())

    def test_reference_matching_and_missing_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root/'dao'/'full'
            dest.mkdir(parents=True)
            row = {'Map name': 'tiny.map', 'Source scenario index': 10,
                   'Start x': 0, 'Start y': 0, 'Goal x': 3, 'Goal y': 0,
                   'A* reference length': 3.0, 'Path length': 3.3, 'Success': True}
            pd.DataFrame([row]).to_csv(dest/'per_run_results_dao.csv', index=False)
            frame = pd.DataFrame([row, dict(row, **{'Goal x': 2})])
            with contextlib.redirect_stdout(io.StringIO()):
                result = reuse_astar_references(frame, {'astar_reference_root': root,
                                                       'dataset_name': 'dao'})
            self.assertAlmostEqual(result.iloc[0]['Gap vs A* percent'], 10)
            self.assertTrue(pd.isna(result.iloc[1]['Gap vs A* percent']))

    def test_journal_signature_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'journal.sqlite3'
            SearchJournal(path, {'w': 1.15}).close()
            with self.assertRaises(ValueError):
                SearchJournal(path, {'w': 1.0})


if __name__ == '__main__':
    unittest.main()
