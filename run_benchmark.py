"""Run the reviewer Weighted JPS baseline; repeat the command to resume."""
import argparse
import subprocess
import sys
from pathlib import Path
from benchmark_config import DATASETS, EXPERIMENT_CONFIGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=sorted(DATASETS))
    parser.add_argument('config', nargs='?', default='full', choices=sorted(EXPERIMENT_CONFIGS))
    parser.add_argument('--config-json', type=Path, default=Path(__file__).with_name('exp_config.json'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, '-u', str(Path(__file__).with_name(
        'Astar_JPS_SA_JPS_config_profiles_ready.py')), args.dataset, args.config,
        '--config-json', str(args.config_json)]
    if args.dry_run:
        command.append('--dry-run')
    raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    main()
