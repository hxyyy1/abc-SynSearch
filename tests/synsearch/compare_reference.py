#!/usr/bin/env python3
"""Optional differential test against a separately supplied SynSearch checkout.

Requires the reference project's NumPy dependency. This is test infrastructure,
not a runtime dependency of the native command. LinUCB policy RNG differences
are reported separately; all other trajectory mismatches fail the comparison.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--abc-bin', type=Path, default=Path(__file__).resolve().parents[2] / 'abc')
    parser.add_argument('--design', type=Path, help='BLIF input; defaults to ABC i10 converted to BLIF')
    parser.add_argument('--seeds', default='0,1,17')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_root.resolve()))
    from alphasyn.backend import ABCBackend
    from alphasyn.mcts import run_search as mcts
    from alphasyn.types import SearchConfig
    from hybridsyn.search import HybridSearchConfig, run_search as hybrid
    from SASyn.backend import ABCBackend as SABackend
    from SASyn.core_types import SearchConfig as SAConfig
    from SASyn.sa import run_search as sa
    abc = args.abc_bin.resolve()
    report = []
    with tempfile.TemporaryDirectory(prefix='synsearch-parity-') as tmp:
        work = Path(tmp)
        design = args.design.resolve() if args.design else work / 'i10.blif'
        if not args.design:
            source = Path(__file__).resolve().parents[2] / 'i10.aig'
            subprocess.run([str(abc), '-s', '-c', f'read {source}; write_blif {design}'], check=True, capture_output=True)
        for seed in map(int, args.seeds.split(',')):
            for method in ('mcts', 'hybrid', 'sa', 'ucb1', 'ucb1-prefix', 'linucb'):
                output = work / f'{method}-{seed}.json'
                flags = ('--steps 3 --search-iterations 3 --episodes 3 --warmup-steps 1 '
                         '--warmup-episodes 3 --warmup-top-k 2 --long-term-horizon 2 --ucb-c 0.4')
                subprocess.run([str(abc), '-s', '-c', f'read_blif {design}; strash; synsearch --algorithm {method} {flags} --seed {seed} --output {output}'], check=True, capture_output=True)
                native = json.loads(output.read_text())
                if method in ('mcts', 'hybrid', 'sa'):
                    kw = dict(design_name=design.name, design_path=design, sequence_length=3,
                              search_iterations=3, seed=seed, workdir=work / f'{method}-{seed}')
                    backend = (SABackend if method == 'sa' else ABCBackend)(str(abc), kw['workdir'])
                    if method == 'mcts':
                        ref = mcts(SearchConfig(**kw), backend)
                    elif method == 'hybrid':
                        ref = hybrid(HybridSearchConfig(**kw, warmup_steps=1, warmup_episodes=3, warmup_top_k=2), backend)
                    else:
                        # Match the reference CLI defaults, which differ from its dataclass defaults.
                        ref = sa(SAConfig(**kw, and_weight=.7, lev_weight=.3), backend)
                    expected = [list(ref.sequence), ref.final_and_count, ref.final_lev_count]
                    actual = [native['sequence'], native['final_and'], native['final_lev']]
                else:
                    name = {'ucb1': 'baseline_mab', 'ucb1-prefix': 'baseline_mab_prefix', 'linucb': 'linucb'}[method]
                    mod = importlib.import_module('MABSyn.' + name)
                    mod.N_EPISODES = 3
                    mod.K_STEPS = 3
                    mod.RANDOM_SEED = seed
                    if method == 'linucb':
                        mod.LONG_TERM_HORIZON = 2
                        ref = mod.optimize_one_benchmark(str(design), abc_bin=str(abc), seed=seed)
                    else:
                        ref = mod.optimize_one_benchmark(str(design), abc_bin=str(abc), bandit_c=.4)
                    expected = [ref['best']['recipe_str'], ref['best']['nodes'], ref['best']['level']]
                    actual = [native['recipe'], native['final_and'], native['final_lev']]
                row = dict(algorithm=method, seed=seed, matches=actual == expected, reference=expected, native=actual)
                report.append(row)
                print(json.dumps(row), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    return int(any(not r['matches'] and r['algorithm'] != 'linucb' for r in report))


if __name__ == '__main__':
    raise SystemExit(main())
