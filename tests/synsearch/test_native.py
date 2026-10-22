"""Integration tests: real ABC networks, native searches, replay and CEC."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
ABC = Path(os.environ.get('ABC_BIN', ROOT / 'abc')).resolve()
METHODS = ('mcts', 'hybrid', 'sa', 'ucb1', 'ucb1-prefix', 'linucb')
BLIF = '''.model example
.inputs a b c d
.outputs y z
.names a b t1
11 1
.names a c t2
11 1
.names t1 t2 y
1- 1
-1 1
.names b c t3
11 1
.names t3 d z
1- 1
-1 1
.end
'''


def run(script):
    return subprocess.run([str(ABC), '-s', '-c', script], capture_output=True, text=True, timeout=120)


class NativeSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synsearch-')
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.design = self.work / 'test.blif'
        self.design.write_text(BLIF)

    def search(self, method, extra=''):
        out = self.work / f'{method}.json'
        aig = self.work / f'{method}.aig'
        command = (f'read_blif {self.design}; strash; synsearch --algorithm {method} '
                   f'--steps 3 --search-iterations 3 --episodes 3 --warmup-steps 1 '
                   f'--warmup-episodes 3 --warmup-top-k 2 --long-term-horizon 2 '
                   f'--debug-search --output {out} {extra}; write_aiger {aig}')
        completed = run(command)
        self.assertTrue(out.exists(), completed.stdout + completed.stderr)
        result = json.loads(out.read_text())
        self.assertTrue(aig.exists(), completed.stdout + completed.stderr)
        self.assertEqual(result['algorithm'], method)
        self.assertGreaterEqual(result['final_and'], 0)
        self.assertLessEqual(len(result['sequence']), 3)
        return result, aig

    def test_all_algorithms_replay_and_equivalence(self):
        for method in METHODS:
            with self.subTest(method=method):
                result, aig = self.search(method)
                replay = self.work / 'replay.aig'
                script = f'read_blif {self.design}; strash; {result["recipe"]}; write_aiger {replay}'
                run(script)
                comparison = run(f'cec {aig} {replay}')
                self.assertIn('Networks are equivalent', comparison.stdout)
                original = run(f'cec -n {self.design} {aig}')
                self.assertIn('Networks are equivalent', original.stdout)

    def test_repeated_seed_and_state_isolation(self):
        for method in METHODS:
            with self.subTest(method=method):
                a, _ = self.search(method)
                b, _ = self.search(method)
                for key in ('sequence', 'final_and', 'final_lev', 'trace'):
                    self.assertEqual(a[key], b[key])

    def test_invalid_options_and_missing_network(self):
        for options in ('--objective invalid', '--algorithm missing', '--steps -1', '--cpuct nan', '--lambda 0',
                        '--actions unknown', '--actions balance,balance', '--steps 1 --sequence-length 2',
                        '--output', '--episodes 0', '--unknown value'):
            out = run(f'read_blif {self.design}; strash; synsearch {options}')
            self.assertNotIn('SynSearch mcts:', out.stdout)
            self.assertTrue('Invalid' in out.stdout + out.stderr or 'Missing' in out.stdout + out.stderr)
        self.assertIn('Read a network', run('synsearch').stdout + run('synsearch').stderr)

    def test_zero_length_and_csv(self):
        csv = self.work / 'summary.csv'
        for method in METHODS:
            result, _ = self.search(method, f'--csv {csv}')
        import csv as csv_module
        with csv.open() as stream:
            self.assertEqual(len(list(csv_module.DictReader(stream))), len(METHODS))
        out = self.work / 'zero.json'
        completed = run(f'read_blif {self.design}; strash; synsearch --steps 0 --output {out}')
        self.assertTrue(out.exists(), completed.stdout + completed.stderr)
        self.assertEqual(json.loads(out.read_text())['sequence'], [])

    def test_reference_trajectories_on_i10(self):
        # Captured from the separately executed reference implementation; see
        # compare_reference.py and docs/PORTING.md for the reproducible setup.
        expected = {
            'mcts': (['rewrite-z', 'balance', 'resub-z'], 1909, 34),
            'hybrid': (['refactor-z', 'rewrite-z', 'balance'], 1936, 32),
            'sa': (['balance', 'resub', 'rewrite'], 1980, 34),
            'ucb1': (['rewrite-z', 'balance', 'resub-z'], 1909, 34),
            'ucb1-prefix': (['balance', 'rewrite', 'rewrite-z'], 1921, 36),
            'linucb': (['refactor', 'rewrite-z', 'rewrite-z'], 1901, 45),
        }
        for method, oracle in expected.items():
            with self.subTest(method=method):
                output = self.work / f'{method}-oracle.json'
                completed = run(f'read {ROOT / "i10.aig"}; strash; synsearch --algorithm {method} '
                                '--steps 3 --search-iterations 3 --episodes 3 --warmup-steps 1 '
                                '--warmup-episodes 3 --warmup-top-k 2 --long-term-horizon 2 '
                                f'--ucb-c 0.4 --output {output}; cec {ROOT / "i10.aig"}')
                self.assertIn('Networks are equivalent', completed.stdout)
                result = json.loads(output.read_text())
                self.assertEqual((result['sequence'], result['final_and'], result['final_lev']), oracle)

    def test_search_preserves_caller_undo_and_failure_state(self):
        output = run(f'read {ROOT / "i10.aig"}; strash; synsearch --steps 1 --search-iterations 1; undo; print_stats')
        self.assertRegex(output.stdout, r'and\s*=\s*2675')
        # Interactive input continues after a command error, allowing inspection
        # of the original frame after an output write fails following evaluation.
        output = subprocess.run([str(ABC), '-s'], input=(
            f'read {ROOT / "i10.aig"}\nstrash\n'
            'synsearch --steps 1 --search-iterations 1 --output /dev/full\nprint_stats\nquit\n'),
            text=True, capture_output=True, timeout=120)
        self.assertIn('original network restored', output.stdout + output.stderr)
        self.assertRegex(output.stdout, r'and\s*=\s*2675')

    def test_single_metric_objectives_match_exhaustive_one_step(self):
        # One-step action enumeration is an independent oracle for the policies,
        # including LinUCB's tail comparison and Hybrid's branch comparison.
        import re
        actions = ('balance', 'rewrite-z', 'refactor')
        observed = []
        for action in actions:
            output = run(f'read {ROOT / "i10.aig"}; strash; {action.replace("-z", " -z")}; print_stats')
            match = re.search(r'and\s*=\s*(\d+).*lev\s*=\s*(\d+)', output.stdout)
            self.assertIsNotNone(match, output.stdout)
            observed.append(tuple(map(int, match.groups())))
        self.assertNotEqual(min(observed)[1], min(x[1] for x in observed),
                            'Oracle must distinguish node and level objectives')
        for objective, column in (('nodes', 0), ('level', 1)):
            for method in METHODS:
                with self.subTest(objective=objective, method=method):
                    output = self.work / f'{objective}-{method}.json'
                    completed = run(f'read {ROOT / "i10.aig"}; strash; synsearch '
                        f'--algorithm {method} --objective {objective} --steps 1 '
                        '--search-iterations 50 --episodes 6 --warmup-steps 1 '
                        '--warmup-episodes 6 --warmup-top-k 3 --long-term-horizon 1 '
                        f'--actions {",".join(actions)} --output {output}; cec {ROOT / "i10.aig"}')
                    self.assertTrue(output.exists(), completed.stdout + completed.stderr)
                    result = json.loads(output.read_text())
                    self.assertEqual(result['objective'], objective)
                    self.assertEqual(result['final_and' if column == 0 else 'final_lev'],
                                     min(x[column] for x in observed))
                    self.assertEqual(result['config']['and_weight'], 1-column)
                    self.assertEqual(result['config']['lev_weight'], column)
                    self.assertEqual(result['config']['level_reward_weight'], column)
                    self.assertIn('Networks are equivalent', completed.stdout)

    def test_current_network_can_be_used_again(self):
        out = run(f'read_blif {self.design}; strash; synsearch --steps 1 --search-iterations 1; '
                  'synsearch --algorithm linucb --steps 1 --episodes 1; print_stats')
        self.assertEqual(out.stdout.count('Recipe:'), 2, out.stdout + out.stderr)


if __name__ == '__main__':
    unittest.main()
