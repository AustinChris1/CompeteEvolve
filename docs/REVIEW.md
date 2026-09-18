# Code review: the Python implementation

Written for the CompeteEvolve authors. Reviewed at commit `a3c3fbc` ("Fixed the
benchmark") on 2026-09-18, Windows 11, Python 3.12.4, scikit-learn 1.6.1. Every
finding below was reproduced, not inferred from reading. Commands are included
so you can re-run them.

## Summary

The architecture matches the paper: agents drive the tools, evolution runs
islands x generations x samples from the fixed original code, a sandbox measures
primary traits, an LLM reviewer counts logical errors, and the reward is derived
from rank. The code is readable and the module boundaries are sensible.

Three problems stand between the current output and a result that supports the
paper's claims, in order of severity:

1. **The fitness signal is dominated by timing noise, not by the algorithm.** P is
   `Ma x N x A / (T x R)`. On Iris, T is a few milliseconds and R is almost constant,
   so P moves by an order of magnitude between two runs of identical code.
2. **The threshold mu is checked against a baseline that is measured per agent,
   per call, while other sandboxes are running.** In the committed traits files,
   3 of 5 agents measured the *original* code as a 60 second timeout, which sets
   P1 = 0 and mu = 1.0. Every agent then reports `reached_threshold: true` after
   one generation. This is the "agents assume the threshold without proof" issue.
3. **The baselines are deliberately crippled**, and the 25 percent target is met by
   restoring scikit-learn defaults. The current results measure "un-breaking a
   configuration", not "improving a scikit-learn algorithm".

Everything else is engineering polish and is listed in section 3.

## 1. Findings that change the results

### 1.1 Timing noise dominates P

The same baseline code, measured by `agent/sandbox.py` five times in a row with
nothing else running, then with 5, 10 and 20 sandboxes running concurrently
(the 5-agent default launches at least 5 at once, and the evolution loop can
launch far more):

| Condition | T median | T min | T max | P* median | P* min | P* max | Errors |
|---|---|---|---|---|---|---|---|
| Sequential, 5 runs | 6.0 ms | 4.4 ms | 37.0 ms | 5.86 | 0.54 | 7.15 | 0 |
| 5 concurrent | 8.0 ms | 7.0 ms | 9.7 ms | 2.54 | 1.92 | 3.23 | 0 |
| 10 concurrent | 9.6 ms | 7.5 ms | 12.0 ms | 2.68 | 1.18 | 4.26 | 0 |
| 20 concurrent | 10.3 ms | 0.0 ms | 223 ms | 2.28 | 0.00 | 69,671,170 | 6 |

Identical code, no LLM involved. P* spans 13x in the sequential case and becomes
meaningless under load. In the committed `benchmarks/logisticregression01.md`
all four working agents have the same Ma (0.9333) and L (0.1740); the ranking
between them was decided by novelty (0.20 vs 0.15 vs 0.13 vs 0.09) and T
(16 ms vs 17 ms vs 23 ms vs 28 ms). Neither says anything about the algorithm.

Two mechanisms:

- T is wall-clock `time.time()` around a fit that takes milliseconds. One
  scheduler hiccup changes it 5x. Nothing is repeated or averaged.
- R = K x Mu is sampled from outside the process by a thread every 20 ms. K is
  dominated by importing pandas and scikit-learn (about 5 CPU seconds in the
  committed reports, versus 0.17 s measured here), so it carries no information
  about the candidate. Under load the sampler thread dies with
  `OSError: [WinError 1455] The paging file is too small`, K and Mu become 0,
  R hits its 1e-6 floor, and P explodes. A crashed sampler wins the competition.

Reproduce: `python scripts/timing_noise.py logistic_regression` (added in this
branch).

A consequence worth stating plainly: because T divides P, a candidate that
scores 0.30 accuracy and reports a 0.2 ms fit outranks one that scores 0.90 in
5 ms (P = 1500 versus 180). The ranking has no accuracy floor. The `proven`
gate added in this branch keeps such a candidate from claiming the threshold,
but it still wins the rank and the reward. Either rank only candidates whose
accuracy is at least the original's, or move T out of the product and into a
constraint (for example, "no slower than the original").

### 1.2 The threshold is checked against an unreliable, per-agent baseline

`evolution.run` re-measures the original code every time any agent calls the
tool, concurrently with everything else. The committed traits files record what
each agent saw as the *original* code's traits during the same run:

| Traits file | Original TT | RE | P (original) |
|---|---|---|---|
| logistic_regression / agent-1 | 60.0 s | 1 | 0 |
| logistic_regression / agent-2 | 60.0 s | 1 | 0 |
| logistic_regression / agent-3 | 60.0 s | 1 | 0 |
| logistic_regression / agent-4 | 0.0998 s | 0 | 0.0101 |
| logistic_regression / agent-5 | 0.1447 s | 0 | 0.0051 |
| knn / agent-1 | 0.1027 s | 0 | 0.0043 |
| knn / agent-3 | 0.5500 s | 0 | 0.0007 |
| knn / agent-4 | 0.1610 s | 0 | 0.0023 |
| knn / agent-5 | 0.2433 s | 0 | 0.0011 |

The same 5-line baseline was a 60 s timeout for three agents and a 0.1 s run for
the other two. For kNN, P1 varies 6x across agents. Consequences:

- With P1 = 0, `compute_mu` returns P2 / (0 + P2) = 1.0, so
  `reached_threshold` is true regardless of what the candidate did.
- The `if mu > mu_threshold: break` check runs at the top of the generation
  loop, so every agent stops after generation 1. All nine committed traits
  files are `gen01`. The paper's Gn = 3 never executes.
- `evaluator.run` measures the baseline again, and `benchmark.run` a third time,
  so the three mu values a run prints are computed against three different P1s.

The fix is to measure the original once, sequentially, before any agent starts,
store it on `SharedState`, and treat "threshold reached" as a claim that
requires evidence: baseline ran cleanly, candidate ran cleanly, P1 > 0, and mu
computed from the shared baseline. Anything else is reported as "not proven".

### 1.3 The baselines are crippled, so the target is trivial

`sklearn_algorithms.py` ships each baseline with hyperparameters chosen to fail
(`max_iter=5, C=0.005` for logistic regression, `n_neighbors=75` for kNN, depth-1
stumps, a sigmoid SVM with `C=0.01`). On the exact split the code uses:

| Algorithm | Repo baseline accuracy | scikit-learn default accuracy |
|---|---|---|
| logistic_regression | 0.7000 | 1.0000 |
| knn | 0.9333 | 1.0000 |
| decision_tree | 0.6333 | 1.0000 |
| random_forest | 0.7000 | 1.0000 |
| svm | 0.3000 | 1.0000 |

Any agent that deletes the crippling arguments gets a 25 percent gain. The
committed winning code for logistic regression is scikit-learn defaults plus
a `StandardScaler` and `stratify=y`. That is a fine result for "the agents
recover a sensible configuration" but it does not support "improving existing
machine learning algorithms". The report needs a second reference row measured
from scikit-learn defaults, and the paper needs to say which of the two it is
improving on.

### 1.4 The test set is 30 rows and the split is evolvable

Iris has 150 rows; `test_size=0.2` leaves 30, so accuracy moves in steps of
0.033 and every agent landed on exactly 28/30. The prompt tells agents the
split "is evolvable", so an agent may change `random_state`. For a fixed good
model, 100 seeds give accuracies from 0.867 to 1.000, and 20 of them give 1.0.
Seed shopping is therefore worth up to 4 accuracy steps, which is more than any
real algorithmic change on this dataset. The logic checker cannot see it. Fix:
the harness owns the split (fixed folds, or repeated stratified k-fold), and the
candidate only provides the model.

### 1.5 Islands are a loop counter

Within a generation the islands share nothing, are seeded from the same
original code, and never migrate. The only difference between island 1 and
island 4 is the `mode` label ("annotate" vs "first"). The annotated output of
island 1 is never fed to island 2; each island re-annotates from scratch. Either
implement islands (separate populations, seeded from their own best, periodic
migration) or drop the word from the paper and call it "samples per generation".

### 1.6 Reward and novelty differ from the paper draft

- `reward_for_rank` gives reward 1 to rank 1 only. The draft says `R >= n/2`.
  Whichever is intended, the feedback prompt tells everyone below rank 1 that
  they are "lagging behind", which for 5 agents is 80 percent of them every time.
- Novelty rewards token dissimilarity from the original. In the reports it is
  the deciding term between agents with identical accuracy. Rewriting comments
  raises it; a better solver does not. Consider novelty as a tie-breaker or a
  diversity filter, not a multiplier in P.

## 2. Defects (crashes and wrong numbers)

| Where | Defect | Effect | Status |
|---|---|---|---|
| `benchmark.py` `run` | `BenchmarkReport(... original=original ...)` but the dataclass has no `original` field | `TypeError` at the end of every run, after the report is written | fixed |
| `sandbox.py` `_sample_resources` | `cpu_seconds = max(cpu_seconds, cpu_seconds + child_cpu...)` adds children's cumulative CPU on every 20 ms tick | K over-counted for any candidate with `n_jobs > 1` | fixed by measuring inside the child |
| `sandbox.py` `_sample_resources` | psutil exceptions other than `psutil.Error` are not caught | thread dies, K = Mu = 0, R = 1e-6, P inflated by 10^6 | fixed |
| `sandbox.py` timeout | `proc.kill()` does not kill grandchildren on Windows | joblib workers survive a timeout and keep loading the machine | fixed |
| `evaluator._fetch_agent_sample` | memory orders by `(cosine_similarity + functional_accuracy) / 2`, not by trait | after two `evolution` calls the evaluator may score the wrong candidate | fixed |
| `traits.compute_novelty` | reference candidate gets N = 1.0, others about 0.15 | the report's original-vs-evolved P column compares different formulas; only P* is comparable | report now uses P* |
| `main.py` | evolution parameters are requested from the LLM in prose ("use generations=3 ...") | the LLM can and does pass other values; the run is not reproducible | enforced from config |
| `agent/__init__.py` `send` | `while True` with no turn cap | an agent that keeps calling tools never ends and spends without bound | capped |
| `llm_client.py` | API key in the URL query string; `resp.json()` before checking status | key leaks into logs and proxies; an HTML 502 raises `JSONDecodeError` | header auth, status checked |
| `llm_client.py` | OpenRouter 402 (credits) is not handled | the kNN report contains a 1 KB error dump instead of annotated code | surfaced as a clear error |
| `memory.py` `run` | a bare `query` string that is not JSON is **written** to memory as code | in the live run both agents opened with `memory {"query": "logistic_regression"}`, storing that word as an agent-less code entry next to the real baseline | bare queries are reads |
| `logic_check.py`, `evolution.py` | exceptions logged as `str(e)`, which is empty for transport errors | the live run printed `review failed ()` and `sample generation failed ()` twelve times with no cause | type and message logged |
| `main.py` | exit code 0 when every agent failed and no report was produced | scripts and CI cannot detect a failed run | non-zero exit |

Observed in a live 2-agent run of the unmodified code (2026-09-18, Gemini):
the two agents measured the same original code at P1 = 1.07 and P1 = 2.11
within seconds of each other, a 2x disagreement before any candidate existed.

## 3. Engineering polish

What "standardized" should mean for this repo, roughly in the order it pays off:

1. **Packaging.** No `pyproject.toml`, no README, no LICENSE (the Rust version had
   one). `prompts.py` lives at the root and three modules do
   `sys.path.insert(0, ...)` to reach it. Ten compiled `.pyc` files for Python
   3.13 were committed. The `.gitignore` had UTF-16 bytes appended to its last
   line. Done in this branch: `pyproject.toml`, `prompts` moved into the
   package, pyc untracked, `.gitignore` rewritten, README added.
2. **Configuration in one place.** Generations, islands, samples, mu threshold,
   sandbox timeout, concurrency, RPM, models and max agent turns were spread over
   constants in `main.py`, `evolution.py`, `traits.py`, env vars and prose in the
   prompt. Now a single `RunConfig` with env overrides, printed at start and
   written into the run directory.
3. **One baseline, measured first.** See 1.2.
4. **Deterministic sandbox measurement.** Sequential by default
   (`SANDBOX_CONCURRENCY=1`), `perf_counter`, K and peak RSS measured inside the
   child at exit instead of sampled from outside, `compile()` in-process instead
   of a `py_compile` subprocess, repeats with median T for reported numbers.
5. **Run artifacts.** Everything a run produces goes to `runs/<run-id>/`:
   config, every sample's code and traits as JSONL, the traits markdown, the
   benchmark report and chart. Nothing is written into the source tree.
6. **Logging instead of print** in library modules, with a `-v` flag. `print`
   stays for the interactive CLI.
7. **Tests and CI.** None existed. Added: trait math (copy has zero novelty,
   mu, accuracy term, rank and reward), memory round trip, sandbox (ok, syntax
   error, runtime error, timeout, kill tree), prompt construction, benchmark
   report construction, and an end-to-end evolution run against a fake LLM
   client that needs no network. CI runs ruff and pytest.
8. **Dead code.** `default_model_for` (with model names that disagree with
   `_ROLE_DEFAULTS`), `remember_csv_file`, the unused `episodes` argument.

## 4. What I would change in the paper

- Report against two references: the crippled baseline (what the system was
  given) and scikit-learn defaults (what a practitioner would start from).
  Claim improvement only where the second is beaten.
- Report T, P and mu as median and spread over at least 5 runs, with the
  sandbox sequential. Single-run P is noise (section 1.1).
- Use a benchmark with enough test rows that accuracy has resolution: repeated
  stratified k-fold on Iris, or more datasets (OMEGA uses 20).
- Say what an "island" is in the implementation, or remove the term.
- State the reward rule that the code implements.

## 5. Reproduction

```bash
pip install -e .[dev]
python scripts/timing_noise.py logistic_regression   # section 1.1
python scripts/baseline_vs_defaults.py               # section 1.3
pytest                                                # unit and fake-LLM tests
python main.py 2 gemini gemini --algorithm logistic_regression   # live run
```
