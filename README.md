# ABC-SynSearch

Native search-based logic synthesis in Berkeley ABC. Six search algorithms are
implemented in C++ and compiled into the `synsearch` ABC command. Candidate
networks are evaluated in the same process and cached in memory; search does not
launch an ABC executable, invoke Python, or write intermediate AIG files.

This project ports the algorithms from SynSearch into an ABC source fork. It
retains their distinct objectives and selection rules. See
[porting and compatibility](docs/PORTING.md) for the exact scope and differences.

## Build

Linux / WSL Ubuntu, GCC or Clang with C++17, GNU Make, and readline development
headers are required for the default build. Python is only needed for optional
experiment tools and integration tests.

```sh
sudo apt-get install build-essential libreadline-dev
make -j8
./abc -s -c 'synsearch --help'
```

A build without readline is supported: `make ABC_USE_NO_READLINE=1 -j8`.
Use a clean build when changing compiler flags.

CMake is also supported:

```sh
cmake -S . -B build -DABC_SKIP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
./build/abc -s -c 'synsearch --help'
```

`ABC_SKIP_TESTS=ON` skips the upstream GoogleTest dependency download; it does
not disable the standalone SynSearch integration tests below.

## Quick start

```sh
./abc -s -c 'read i10.aig; strash; synsearch --algorithm mcts --steps 4 --search-iterations 4 --output result.json; cec i10.aig; write_aiger optimized.aig'
```

The command operates on the current strashed network and installs the selected
network on success. `print_stats`, `cec`, `write_blif`, and `write_aiger` can be
used afterward. A failed required evaluation restores the original network.
Use `cec -n input.blif optimized.aig` when AIG serialization has omitted signal
names and matching by input/output order is appropriate.

| Algorithm | `--algorithm` | Native behavior |
| --- | --- | --- |
| AlphaSyn | `mcts` | Uniform-prior MCTS, discounted max-return backup, subtree reuse |
| HybridSyn | `hybrid` | Per-position UCB1 warmup, top-k prefix handoff, MCTS continuation |
| SASyn | `sa` | Full-sequence annealing, replace/swap/block moves, automatic temperature |
| Global UCB1 | `ucb1` | One bandit, complete-episode reward, unique-arm updates |
| Prefix UCB1 | `ucb1-prefix` | One bandit per position, immediate rewards, cached prefixes |
| LinUCB | `linucb` | Per-position contextual policies, rollout features, tail selection and backtracking |

```sh
./abc -s -c 'read i10.aig; strash; synsearch --algorithm hybrid --steps 10 --warmup-steps 3 --warmup-episodes 20 --warmup-top-k 2 --search-iterations 20'
./abc -s -c 'read i10.aig; strash; synsearch --algorithm sa --steps 10 --search-iterations 100 --seed 0'
./abc -s -c 'read i10.aig; strash; synsearch --algorithm ucb1 --steps 10 --episodes 20 --ucb-c 1.0'
./abc -s -c 'read i10.aig; strash; synsearch --algorithm ucb1-prefix --steps 10 --episodes 20 --ucb-c 0.4'
./abc -s -c 'read i10.aig; strash; synsearch --algorithm linucb --steps 10 --episodes 20 --alpha 1.5 --lambda 1 --long-term-rollouts 1 --long-term-horizon 3'
```

### Single-metric experiments

`--objective nodes` minimizes AIG node count; `--objective level` minimizes
logic depth. The default `--objective legacy` preserves the original algorithm
semantics documented in `docs/PORTING.md`. Explicit objectives override
`--and-weight`, `--lev-weight`, and `--level-reward-weight`; they also switch
MCTS/Hybrid reward and branch comparisons, plus LinUCB tail/depth-best comparisons
and its backtrack gap. LinUCB uses the other metric only to break primary-metric
ties. The existing node-spike guard remains active. These modes adapt the
project's policies to single-metric tasks, rather than reproducing the reference
algorithms' original mixed objectives.

```sh
./abc -s -c 'read_blif benchmarks/VTR/bfly.abc.blif; strash; synsearch --algorithm mcts --objective level --steps 10 --output level.json'
python3 tools/synsearch/vtr_objectives.py --out /path/to/fresh/results --workers 3
```

The VTR driver runs six methods, two objectives and seeds 0/1/2, records original
and resyn2 baselines, pins workers to distinct physical cores, and verifies
combinational equivalence across unchanged latch boundaries. See its manifest
for explicit method budgets; equal sequence lengths do not mean equal compute.
Search and CEC have separate timeouts. Completed run folders can be resumed
only when the recorded source, binary, driver and experiment settings match.

To export comparison tables, install `python-docx` and `XlsxWriter` in your Python
environment, then run:

```sh
python3 tools/synsearch/summarize_vtr_objectives.py /path/to/results --out /path/to/tables
```

The exports include both best-of-three and mean/standard-deviation results,
per-run CSV records, an Excel workbook, and a Word experiment report. All
expected runs and their CEC checks must succeed before the final tables are
produced. The memory-constrained VTR configuration uses three workers by default;
size concurrency for the account's cgroup limit, not just the machine's free RAM.

### Parameters and actions

`--steps` and `--sequence-length` are aliases. MCTS iterations apply at every
selected step; SA iterations count total annealing proposals; bandit methods use
`--episodes`. Equal numerical budgets do not imply equal work.

Default lengths are 24 for MCTS/HybridSyn/SA and 10 for the bandit methods.
MCTS uses 64 iterations, SA uses 1000 iterations, UCB1 uses 50 episodes, and LinUCB uses 150 episodes. Global UCB1 defaults to `--ucb-c 2.0`; prefix UCB1 and HybridSyn use 0.4.
Hybrid warmup defaults to 4 steps, 64 episodes, and one retained candidate.

The default actions are `balance`, `rewrite`, `rewrite-z`, `refactor`,
`refactor-z`, `resub`, and `resub-z`. Extra supported action labels are `fraig`,
`fx`, `mfs`, `dsd`, `dch`, `extract`, `extract-a`, and `collapse`. Labels with `-z`
expand to the corresponding ABC command with the `-z` flag. Some extra operators
have network-type preconditions and can fail; they are not silently substituted.

```sh
./abc -s -c 'read i10.aig; strash; synsearch --actions balance,rewrite,rewrite-z --steps 4 --search-iterations 4 --debug-search --output result.json --csv summary.csv'
```

`--help` lists all controls, including objective weights, exploration constants,
annealing temperatures, rollout limits, convergence streak, and backtrack threshold.
The search is deterministic for a fixed native build, input, options, and seed.
LinUCB random tie-breaking is not bit-for-bit NumPy compatible.

## Results and experiments

`--output` writes a JSON result containing the configuration, action labels,
expanded commands, initial/final AND and level counts, runtime, evaluation/cache
counters, and optional event trace. `--csv` appends one summary row, adding the
header to an empty file. Parent output directories must already exist.
AND count and logic depth are structural AIG metrics, not mapped area or timing.

The optional Python 3.10+ driver orchestrates complete native runs. Search policy
and candidate evaluations remain inside ABC. It does not require NumPy.
The original SynSearch dataset is included under `benchmarks/`, preserving its
EPFL and VTR directory structure and all supplied circuit representations.
The driver recursively reads BLIF files only: 20 from `benchmarks/EPFL/` and
6 from `benchmarks/VTR/`. Its default dataset path is resolved relative to this
repository, so it also works when invoked from another working directory.
Use `--dataset-root benchmarks/EPFL` or `--dataset-root benchmarks/VTR` to select
a subset when running from the repository root; custom paths remain supported.
These circuit files were copied unchanged from the original project's dataset;
the source-code license does not relicense third-party benchmarks.

```sh
python3 tools/synsearch/experiments.py prepare-data
python3 tools/synsearch/experiments.py run-search --design adder.blif --algorithm all --sequence-length 4 --search-iterations 4 --episodes 4 --external-monitor
python3 tools/synsearch/experiments.py summarize
```

The driver writes a SHA-256 manifest, per-run JSON/log/AIG files, and per-method
CSV summaries under `.synsearch_work/`. It refuses to overwrite an existing run
folder. Choose a fresh `--workdir` for a new experiment. `--external-monitor`
uses GNU `time` to measure the whole ABC process and peak RSS in KiB; native
JSON runtime otherwise measures search only.

Additional workflows:

```sh
# grid.json: {"ucb-c": [0.2, 0.4, 1.0], "episodes": [10, 20]}
python3 tools/synsearch/experiments.py grid-search --algorithm ucb1 --grid-config grid.json --workdir .synsearch_work/grid
# Baseline plus one omitted action per trial
python3 tools/synsearch/experiments.py ablation --workdir .synsearch_work/ablation
# One-step action measurements; --result-path also evaluates saved search prefixes
python3 tools/synsearch/experiments.py evaluate-actions --workdir .synsearch_work/actions
```

`--config-json` accepts an object of native command options without leading `--`.
The comparison utility is retained from SynSearch and accepts the generated CSVs:

```sh
python3 tools/compare_algorithm_summaries.py --summary MCTS=.synsearch_work/trial-0/mcts-summary.csv --summary UCB1=.synsearch_work/trial-0/ucb1-summary.csv --print-report
```

Use external monitoring consistently when comparing resource metrics. Keep
circuit sets, builds, machine, seeds, and budgets in your experiment records.

## Test and contribute

```sh
python3 -m unittest discover -s tests/synsearch -v
# To test a CMake build:
ABC_BIN="$PWD/build/abc" python3 -m unittest discover -s tests/synsearch -v
```

Tests exercise all six algorithms, deterministic replay, repeated searches,
invalid options, zero-length search, JSON/CSV export, and CEC equivalence.
[CONTRIBUTING.md](CONTRIBUTING.md) explains development and reporting requirements.

## License and provenance

ABC retains its University of California license in [copyright.txt](copyright.txt).
SynSearch additions are available under the MIT license in [LICENSE](LICENSE).
Bundled upstream components keep their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No third-party benchmark license
is implied by the source-code license.
