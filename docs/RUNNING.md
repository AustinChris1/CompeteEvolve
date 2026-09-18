# Running CompeteEvolve

## Setup

```bash
pip install -e .[dev]
cp .env.example .env     # set GEMINI_API_KEY and/or OPENROUTER_API_KEY
```

Python 3.10 or newer. Candidates run in a subprocess started with `python` on
Windows and `python3` elsewhere; set `SANDBOX_PYTHON` if that interpreter is
not the one with scikit-learn installed.

## Commands

```bash
python main.py                                           # fully interactive
python main.py 5 gemini gemini --algorithm logistic_regression
python main.py 3 gemini --algorithm knn --generations 2 --islands 2 --samples 3 --rpm 8
python main.py 2 openrouter --code path/to/algo.py --data path/to/data.csv
python main.py 2 gemini --algorithm svm --no-logic-check -v
```

Positional arguments: number of agents, chat provider, evolution provider.
When the evolution provider is omitted it is the same as the chat provider.
Anything omitted is asked for interactively, except `--algorithm`, which
skips the algorithm menu.

| Flag | Meaning |
|---|---|
| `--algorithm NAME` | `logistic_regression`, `knn`, `decision_tree`, `random_forest`, `svm` |
| `--code FILE --data FILE` | your own baseline (`def run(data_path) -> dict`) and CSV |
| `--generations N`, `--islands N`, `--samples N` | caps for every `evolution` call in the run |
| `--rpm N` | LLM requests per minute, shared by all agents and tools |
| `--no-logic-check` | skip the LLM review that counts logical errors (EL) |
| `--run-id ID` | name of the directory under `runs/` |
| `-v` | debug logging with module names |

## What a run prints

1. The LLM roles, the algorithm and dataset, the run directory and the resolved config.
2. The original code measured once (`original Ma=... TT=... P*=...`) and, for
   built-in algorithms, scikit-learn defaults measured the same way (`reference`).
   If the original fails in the sandbox the run stops with exit code 2.
3. Per agent: each tool call, and inside `evolution` one line per sample with
   Ma, L, T, N, SE, RE, EL and T0. After each generation: the best T0, mu, and
   whether the threshold is proven.
4. Each agent's closing text.
5. The benchmark report: ranking table, best agent, original vs reference vs
   evolved on P*, and where the report, chart and summary were written.

Exit codes: 0 success, 1 no agent completed an evaluation, 2 configuration or
input error.

## Reading `threshold`

Every `evolution` and `evaluator` result carries a `threshold` object:

```json
{"mu": 0.71, "threshold": 0.5556, "performance_original": 1.9,
 "performance_candidate": 4.7, "reached": true, "proven": true, "reasons": []}
```

`reached` is the bare inequality mu > threshold. `proven` additionally requires
that the original ran cleanly with P1 > 0, that the candidate has no syntax,
runtime or logical errors, and that its accuracy is not below the original's.
`reached_threshold` in the tool result equals `proven`; an agent is never told
it succeeded on a number that a crash or a lucky timing produced.

## Output files

`runs/<run-id>/`:

| File | Contents |
|---|---|
| `config.json` | resolved config, model roles, algorithm, dataset |
| `samples.jsonl` | every evaluated candidate: primary and secondary traits, mode, stderr tail, code |
| `traits/<algorithm>/<agent>_genNN_best_traits.md` | top three per agent per generation, as given to the crossover prompt |
| `<algorithm>NN.md`, `<algorithm>NN.png` | copy of the benchmark report and chart |
| `summary.json` | ranking, threshold evidence, file paths |

`benchmarks/<algorithm>NN.md` is the versioned report kept in the repository.

## Configuration reference

| Setting | Default | Env | Notes |
|---|---|---|---|
| `agents` | 5 | `CE_AGENTS` | |
| `generations` | 3 | `CE_GENERATIONS` | an agent may request fewer, never more |
| `islands` | 4 | `CE_ISLANDS` | |
| `samples_per_island` | 5 | `CE_SAMPLES_PER_ISLAND` | |
| `mu_threshold` | 0.5556 | `CE_MU_THRESHOLD` | 1.25 / 2.25, a 25 percent improvement |
| `max_agent_turns` | 12 | `CE_MAX_AGENT_TURNS` | LLM turns per agent before the loop ends |
| `sandbox_timeout` | 60 | `CE_SANDBOX_TIMEOUT` | seconds per candidate execution |
| `sandbox_concurrency` | 1 | `CE_SANDBOX_CONCURRENCY` | keep at 1 for reported numbers |
| `sandbox_repeats` | 1 | `CE_SANDBOX_REPEATS` | median wall time over N executions |
| `llm_rpm` | 10 | `CE_LLM_RPM` | free Gemini tier is about 10 |
| `logic_check` | on | `CE_LOGIC_CHECK` | |
| chat model | provider default | `CHAT_MODEL` | |
| evolution model | provider default | `EVOLUTION_MODEL` | |

## Budget

Each `evolution` call makes `islands x samples` generation requests plus one
logic-check request per sample that ran. With the defaults that is 40 requests
per generation per agent, 120 per agent for three generations, 600 for five
agents, before the evaluator and reinforcement calls. At 10 requests per
minute a default run needs at least an hour of LLM time. Lower `--islands`
and `--samples` for a smoke test.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `GEMINI_API_KEY not set` | create `.env` from `.env.example` |
| `could not launch a Python interpreter` | set `SANDBOX_PYTHON` to a full path |
| `the original code failed in the sandbox` | the baseline itself is broken; see the stderr printed with it |
| repeated `HTTP 429` warnings | lower `--rpm`; a few per run are normal |
| `openrouter has no credits left` | the chosen OpenRouter model is not free for your account |
| P values differ wildly between runs | keep `sandbox_concurrency` at 1 and raise `sandbox_repeats` |
