#!/usr/bin/env python3
"""Experiment orchestration only; every search runs inside the compiled ABC.

No search policy is implemented in Python. One process is used per experiment,
not per candidate evaluation. GNU time optionally measures the complete process.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time

METHODS = ('mcts', 'hybrid', 'sa', 'ucb1', 'ucb1-prefix', 'linucb')
DEFAULT_ACTIONS = 'balance,rewrite,rewrite-z,refactor,refactor-z,resub,resub-z'


def quoted(value):
    value = str(value)
    if any(c in value for c in '\n\r";'):
        raise ValueError('ABC paths/options cannot contain quotes, semicolons or newlines')
    return '"' + value + '"'


def designs(root, selected):
    files = sorted(root.resolve().rglob('*.blif'))
    if selected:
        files = [p for p in files if p.name in selected or p.relative_to(root.resolve()).as_posix() in selected]
    if not files:
        raise ValueError('No matching BLIF designs')
    names = [p.name for p in files]
    return [(p.name if names.count(p.name) == 1 else p.relative_to(root.resolve()).as_posix(), p) for p in files]


def invoke(abc, design, name, method, options, folder, monitor, prefix=''):
    folder.mkdir(parents=True, exist_ok=False)
    output = (folder / 'result.json').resolve()
    arguments = ' '.join('--' + k + ' ' + quoted(v) for k, v in options.items())
    script = (f'read_blif {quoted(design)}; strash; {prefix}'
              f'synsearch --algorithm {method} {arguments} --output {quoted(output)}; '
              f'write_aiger {quoted((folder / "optimized.aig").resolve())}')
    command = [str(abc), '-s', '-c', script]
    metrics = folder / 'resources.txt'
    if monitor:
        if not Path('/usr/bin/time').exists():
            raise ValueError('External monitoring requires GNU /usr/bin/time')
        command = ['/usr/bin/time', '-f', '%e %M', '-o', str(metrics), *command]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True)
    (folder / 'abc.log').write_text(completed.stdout + completed.stderr)
    if completed.returncode or not output.exists() or not (folder / 'optimized.aig').exists():
        raise RuntimeError(f'Native search failed: see {folder / "abc.log"}')
    result = json.loads(output.read_text())
    result.update(design_name=name, design_path=str(design), wall_runtime_sec=time.perf_counter()-started)
    if monitor:
        wall, memory = metrics.read_text().split()[-2:]
        result.update(total_runtime_sec=float(wall), peak_memory_kb=float(memory))
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


def summarize(root):
    results = [(p.parent.parent.parent, json.loads(p.read_text())) for p in sorted(root.rglob('result.json'))]
    fields = ['design_name', 'seed', 'initial_and', 'initial_lev', 'final_and', 'final_lev',
              'total_runtime_sec', 'peak_memory_kb', 'recipe']
    groups = {}
    for trial, result in results:
        groups.setdefault((trial, result['algorithm']), []).append(result)
    for (trial, method), rows in groups.items():
        output = trial / f'{method}-summary.csv'
        with output.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare-data', 'run-search', 'summarize', 'grid-search', 'ablation', 'evaluate-actions'))
    parser.add_argument('--abc-bin', type=Path, default=Path(__file__).resolve().parents[2] / 'abc')
    parser.add_argument('--dataset-root', type=Path, default=Path('benchmarks'))
    parser.add_argument('--design', action='append')
    parser.add_argument('--workdir', type=Path, default=Path('.synsearch_work'))
    parser.add_argument('--algorithm', choices=(*METHODS, 'all'), default='mcts')
    parser.add_argument('--sequence-length', type=int, default=4)
    parser.add_argument('--search-iterations', type=int, default=4)
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--actions', default=DEFAULT_ACTIONS)
    parser.add_argument('--external-monitor', action='store_true')
    parser.add_argument('--config-json', type=Path, help='JSON object of native options, without leading --')
    parser.add_argument('--grid-config', type=Path, help='JSON object mapping native options to lists of values')
    parser.add_argument('--result-path', type=Path, help='Evaluate actions after each prefix of this native result')
    args = parser.parse_args()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == 'summarize':
        summarize(root)
        return
    if args.result_path:
        prior = json.loads(args.result_path.read_text())
        files = [(prior['design_name'], Path(prior['design_path']))]
    else:
        files = designs(args.dataset_root, args.design)
    manifest = [{'name': name, 'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for name, p in files]
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.command == 'prepare-data':
        return
    base = {'sequence-length': args.sequence_length, 'search-iterations': args.search_iterations,
            'episodes': args.episodes, 'seed': args.seed, 'actions': args.actions}
    if args.config_json:
        base.update(json.loads(args.config_json.read_text()))
    forbidden = {'algorithm', 'output', 'csv', 'steps'}
    if forbidden.intersection(base):
        parser.error('Use driver options for algorithm/output and sequence-length for the budget')
    if any(not k.replace('-', '').isalnum() for k in base):
        parser.error('Invalid native option name')
    configs = [base]
    if args.command == 'grid-search':
        if not args.grid_config:
            parser.error('grid-search requires --grid-config')
        grid = json.loads(args.grid_config.read_text())
        if forbidden.intersection(grid) or any(not k.replace('-', '').isalnum() for k in grid):
            parser.error('Invalid or reserved grid option')
        if any(not isinstance(v, list) or not v for v in grid.values()):
            parser.error('Each grid option must have a nonempty value list')
        configs = [dict(base, **dict(zip(grid, values))) for values in itertools.product(*grid.values())]
    if args.command == 'ablation':
        labels = base['actions'].split(',')
        if len(labels) < 2:
            parser.error('ablation requires at least two actions')
        configs += [dict(base, actions=','.join(a for a in labels if a != removed)) for removed in labels]
    prefixes = ['']
    if args.result_path:
        prior = json.loads(args.result_path.read_text())
        recipe = prior['commands']
        prefixes = ['; '.join(recipe[:i]) + ('; ' if i else '') for i in range(len(recipe)+1)]
        files = [(prior['design_name'], Path(prior['design_path']))]
    if args.command == 'evaluate-actions':
        configs = [dict(base, actions=action, **{'sequence-length': 1, 'search-iterations': 1}) for action in base['actions'].split(',')]
    methods = METHODS if args.algorithm == 'all' else [args.algorithm]
    if args.command == 'evaluate-actions':
        methods = ['mcts']
    for trial, config in enumerate(configs):
        for method in methods:
            for name, design in files:
                key = hashlib.sha256(name.encode()).hexdigest()[:12]
                for depth, prefix in enumerate(prefixes):
                    folder = root / f'trial-{trial}' / method / f'{key}-prefix-{depth}'
                    context_name = f"{name} [prefix={depth}]" if len(prefixes) > 1 else name
                    result = invoke(args.abc_bin.resolve(), design, context_name, method, config, folder, args.external_monitor, prefix)
                    print(method, name, result['final_and'], result['final_lev'])
        summarize(root / f'trial-{trial}')


if __name__ == '__main__':
    main()
