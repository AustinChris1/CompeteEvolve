# CompeteEvolve

Competing LLM agents that improve machine-learning code through evolutionary
search. Each agent drives four tools: `memory` (the shared store), `evolution`
(islands x generations x samples of candidate code, measured in a sandbox),
`evaluator` (scores the agent's best candidate and ranks it against the others)
and `reinforcement` (reward and the top agents' trait tables). The winner is the
agent whose candidate has the highest trait score T0 = P.

Documentation:

- [docs/REVIEW.md](docs/REVIEW.md): the engineering review of the implementation, with reproducible evidence.
- [docs/RUNNING.md](docs/RUNNING.md): setup, every flag and setting, what a run prints and writes.

## Install

```bash
pip install -e .[dev]
cp .env.example .env          # GEMINI_API_KEY and/or OPENROUTER_API_KEY
```

Python 3.10 or newer. The sandbox launches candidates with the interpreter on
`PATH` (`python` on Windows, `python3` elsewhere); set `SANDBOX_PYTHON` to
override.

## Run

```bash
python main.py                                   # interactive: agents, providers, algorithm
python main.py 5 gemini gemini --algorithm knn    # non-interactive
python main.py 3 gemini --algorithm svm --generations 2 --islands 2 --samples 3 --rpm 8
python main.py 2 openrouter --code my_algo.py --data my_data.csv
```

Every run writes to `runs/<algorithm>-<timestamp>/`:

| File | Contents |
|---|---|
| `config.json` | the resolved configuration, models and inputs |
| `samples.jsonl` | one line per evaluated candidate: traits, mode, code |
| `traits/<algorithm>/<agent>_genNN_best_traits.md` | each agent's top three per generation |
| `<algorithm>NN.md`, `.png` | copy of the benchmark report and chart |
| `summary.json` | ranking, threshold evidence, paths |

The benchmark report is also versioned into `benchmarks/`.

## How a run proceeds

1. The original code (CO) and the dataset are seeded into memory.
2. CO is measured **once, sequentially**, before any agent starts. Every mu in
   the run is computed against this one measurement. For built-in algorithms
   the same harness is also run with scikit-learn's default hyperparameters,
   and that reference appears in the report next to the baseline.
3. Agents run concurrently. Their `evolution` calls are capped by the run
   configuration (an agent may ask for fewer generations, never more).
4. "Threshold reached" is reported only when it is proven: the baseline and
   the candidate both ran cleanly, P1 > 0, the candidate's accuracy is not
   below the original's, and mu > 0.5556. Otherwise the tool returns the
   reasons and the agent is told to keep evolving.
5. The report ranks agents, annotates the winning code, and compares the
   original, the scikit-learn reference and the evolved code on P*
   (performance with novelty factored out, the only column comparable across
   the three).

## Configuration

Defaults, environment variables and flags, in that order of precedence:

| Setting | Default | Env | Flag |
|---|---|---|---|
| agents | 5 | `CE_AGENTS` | positional |
| generations | 3 | `CE_GENERATIONS` | `--generations` |
| islands | 4 | `CE_ISLANDS` | `--islands` |
| samples per island | 5 | `CE_SAMPLES_PER_ISLAND` | `--samples` |
| mu threshold | 0.5556 | `CE_MU_THRESHOLD` | |
| max agent turns | 12 | `CE_MAX_AGENT_TURNS` | |
| sandbox timeout (s) | 60 | `CE_SANDBOX_TIMEOUT` | |
| sandbox concurrency | 1 | `CE_SANDBOX_CONCURRENCY` | |
| sandbox repeats | 1 | `CE_SANDBOX_REPEATS` | |
| LLM requests per minute | 10 | `CE_LLM_RPM` | `--rpm` |
| logic check | on | `CE_LOGIC_CHECK` | `--no-logic-check` |
| models | provider defaults | `CHAT_MODEL`, `EVOLUTION_MODEL` | |

Keep sandbox concurrency at 1 for any number you intend to report; see
[docs/REVIEW.md](docs/REVIEW.md) section 1.1 for what happens otherwise.
Use `CE_SANDBOX_REPEATS=3` or more for published runs.

## Tests

```bash
ruff check .
pytest -q
python scripts/timing_noise.py logistic_regression
python scripts/baseline_vs_defaults.py
```

The tests need no API key. The pipeline test runs evolution, the evaluator,
reinforcement and the benchmark end to end against a fake LLM.
