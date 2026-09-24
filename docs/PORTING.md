# Porting contract

## Source basis

This port was developed against SynSearch commit
`2791b089d8019d45b77467d7b18e947d6ce8f574` and ABC commit
`43d923fbb53f7297f218167ac3282d7c251a67bf`.

| Reference source | Native implementation |
| --- | --- |
| `alphasyn/mcts.py`, `alphasyn/backend.py` | `mcts`, `expand`, `recompute`, `Engine::baseline` |
| `hybridsyn/search.py` | `hybrid` |
| `SASyn/sa.py` | `anneal`, `neighbor` |
| `MABSyn/baseline_mab.py` | `Bandit`, `ucb(..., false)` |
| `MABSyn/baseline_mab_prefix.py` | `ucb(..., true)` |
| `MABSyn/linucb.py` | `LinearArm`, `linucb` |

All native algorithms reside in `src/extSynSearch/synsearch.cpp`. They use ABC's
in-process command dispatcher for operators and own duplicated `Abc_Ntk_t`
networks with RAII. No search policy is executed by an external runtime.

## Preserved decisions

- MCTS expands all actions, uses uniform priors and lexicographically greatest
  action labels on exact score ties. Selection is `reward + child_value + U`.
  Node value is `mu * max(reward + child_value)`, not an average-return backup.
  Only traversed edges receive visits. The selected subtree survives each step.
- MCTS and HybridSyn normalize signed square-root improvements using resyn2's
  AND/depth deltas, with the original fallback baselines and 5:2 weights.
- HybridSyn uses independent per-position UCB1 policies, keeps distinct top-k
  prefixes by weighted cost and lexicographic ties, and compares completed branches.
- SA repeats the allowed resyn2 seed, samples replace/swap/block moves with
  probabilities 0.6/0.2/0.2, estimates temperature from 32 proposals when omitted,
  uses geometric cooling and Metropolis acceptance, and retains the best energy.
- Global UCB1 increments counts while constructing the full recipe and updates
  each distinct chosen arm once using the complete-episode reward.
- Prefix UCB1 updates each position using immediate relative improvement.
  Both UCB methods retain only strictly better weighted-cost completed recipes.
- LinUCB uses ten features: bias, normalized node/depth/average-fanout values,
  and mean/max/population-standard-deviation of node/depth rollout gains. The
  short context describes the candidate's resulting network. Policies are per
  position. Rewards default to relative node reduction; rollout best states
  compare nodes then depth. Tail finalization, same-arm convergence, one-time
  per-position backtracking, exclusions, and policy resets are retained. The
  returned network follows the original final trajectory, not its best-ever prefix.

## Deliberate interface and execution differences

This is a native command migration, not a drop-in replacement for the Python CLI.

- Read circuits with normal ABC commands and run `strash` before `synsearch`.
  `--abc-bin`, Python module entry points, and disk-cache workdirs are not native
  command options. The optional experiment driver supplies batch workflows.
- Native snapshots live in RAM rather than temporary AIG files. Memory scales
  with unique evaluated prefixes; there is no automatic cache eviction.
- JSON schema version 1 and compact event traces are native formats. They are
  not identical to each original algorithm's Python result/debug schema. Common
  summary columns are compatible with the retained comparison utility.
- CPython-compatible integer-seeded MT19937 draws are used for UCB1, HybridSyn,
  SA, and LinUCB rollouts. LinUCB policy tie-breaking uses standard MT19937 rather
  than NumPy PCG64; fixed seeds do not guarantee identical LinUCB trajectories.
- LinUCB uses Sherman–Morrison inverse updates rather than repeatedly inverting
  its matrix. These are algebraically equivalent but can differ in floating-point
  rounding near score ties. The original `pfan` two-decimal context is retained.
- In-process operators cannot safely be killed individually. The old subprocess
  timeout option is not supported. Apply an external process timeout to a whole
  experiment if required. Native assertions/crashes also affect the whole run.
- Resource monitoring measures complete experiments, not each candidate process.
  This avoids pretending that the old subprocess memory/time costs still apply.
- Named built-in actions are accepted. Arbitrary shell commands or arbitrary ABC
  scripts are not accepted as search actions. Operators must leave a strashed
  network to be valid candidates.
- Zero-length search is supported. Invalid/non-finite arguments, empty action
  spaces and duplicate actions are rejected. LinUCB requires positive rollout
  count/horizon because its reference tail selection needs rollout samples.

These differences mean algorithmic fidelity is testable, but byte-identical
output files, performance measurements, and every seeded trajectory are not
promised. CEC checks functional circuit equivalence separately from search-policy
conformance.

## Validation summary

Make and CMake Release builds passed, along with ten integration tests covering
all six algorithms, CEC equivalence, replay, invalid arguments, output failures,
caller undo preservation, batch runs, monitoring, comparison, grid search and
static/prefix action evaluation.

Reference comparisons matched the selected sequence, AND count and level count
in 24 cases: all six algorithms on `i10` with seeds 0, 1 and 17, and on EPFL
`adder` with seed 0. These used length 3, three iterations/episodes, HybridSyn
warmup length 1 / three episodes / top-k 2, LinUCB horizon 2 and UCB coefficient
0.4. SA used its CLI weights of 0.7 / 0.3. CEC also passed for all six methods
on EPFL `adder` and `ctrl`. These checks do not establish parity for all inputs.

Reproduce differential checks with the separate reference checkout and its
NumPy dependency:

```sh
python3 tests/synsearch/compare_reference.py --reference-root /path/to/SynSearch
```

The comparison script fails on non-LinUCB trajectory mismatches and reports
LinUCB mismatches separately because of the implementation differences above.
