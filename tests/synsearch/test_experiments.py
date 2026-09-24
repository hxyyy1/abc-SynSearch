import json
from pathlib import Path
import subprocess
import sys
import unittest
from test_native import ABC, ROOT
import test_native


class ExperimentTests(unittest.TestCase):
    setUp = test_native.NativeSearchTests.setUp
    def driver(self, command, *extra):
        work = self.work / command
        result = subprocess.run([
            sys.executable, str(ROOT / 'tools/synsearch/experiments.py'), command,
            '--abc-bin', str(ABC), '--dataset-root', str(self.work), '--design', 'test.blif',
            '--workdir', str(work), '--sequence-length', '1', '--search-iterations', '1',
            '--episodes', '1', '--actions', 'balance,rewrite', *extra,
        ], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return work

    def test_manifest_and_all_native_methods(self):
        work = self.driver('prepare-data')
        manifest = json.loads((work / 'manifest.json').read_text())
        self.assertEqual(len(manifest), 1)
        self.assertEqual(len(manifest[0]['sha256']), 64)
        work = self.driver('run-search', '--algorithm', 'all', '--external-monitor')
        self.assertEqual(len(list(work.rglob('result.json'))), 6)
        for output in work.rglob('result.json'):
            self.assertGreater(json.loads(output.read_text())['peak_memory_kb'], 0)
        self.assertEqual(len(list(work.rglob('*-summary.csv'))), 6)
        comparison = subprocess.run([
            sys.executable, str(ROOT / 'tools/compare_algorithm_summaries.py'),
            '--summary', f'MCTS={work / "trial-0/mcts-summary.csv"}',
            '--summary', f'UCB1={work / "trial-0/ucb1-summary.csv"}',
            '--output', str(self.work / 'aggregate.csv'),
            '--details-output', str(self.work / 'details.csv')], capture_output=True, text=True)
        self.assertEqual(comparison.returncode, 0, comparison.stdout + comparison.stderr)

    def test_grid_ablation_and_action_evaluation(self):
        grid = self.work / 'grid.json'
        grid.write_text('{"ucb-c": [0.2, 0.4]}')
        work = self.driver('grid-search', '--algorithm', 'ucb1', '--grid-config', str(grid))
        self.assertEqual(len(list(work.rglob('result.json'))), 2)
        work = self.driver('ablation')
        self.assertEqual(len(list(work.rglob('result.json'))), 3)
        work = self.driver('evaluate-actions')
        self.assertEqual(len(list(work.rglob('result.json'))), 2)


    def test_saved_prefix_action_evaluation(self):
        work = self.driver('run-search')
        result = next(work.rglob('result.json'))
        work = self.driver('evaluate-actions', '--result-path', str(result))
        self.assertEqual(len(list(work.rglob('result.json'))), 4)
