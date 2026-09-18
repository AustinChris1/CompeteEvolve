from agent import traits
from agent.traits import PrimaryTraits


def _ts(accuracy=0.9, time_=0.01, se=0, re=0, el=0, code="x = 1"):
    p = PrimaryTraits(training_time=time_, inference_time=0.0, syntax_errors=se, runtime_errors=re,
                      logical_errors=el, loss=0.2, model_accuracy=accuracy, memory_usage=100.0, compute_cost=0.5)
    return traits.build(p, code, "original code", None)


def test_identical_code_has_zero_novelty():
    assert traits.compute_novelty("def run(): pass", "def run(): pass", None) == 0.0
    assert traits.compute_novelty("def run(): pass", "something else entirely", None) > 0.5


def test_accuracy_term_penalises_errors():
    assert traits.compute_accuracy_term(0, 0, 0) == 1.0
    assert traits.compute_accuracy_term(1, 0, 0) == 0.5
    assert traits.compute_accuracy_term(0, 0, 1) < 0.4


def test_mu_and_rank_and_reward():
    assert abs(traits.compute_mu(1.0, 1.25) - 1.25 / 2.25) < 1e-12
    assert traits.compute_mu(0.0, 0.0) == 0.0
    positions = dict(traits.rank_positions([("a", 0.5), ("b", 0.9), ("c", 0.1)]))
    assert positions == {"a": 2, "b": 1, "c": 3}
    assert traits.reward_for_rank(1) == 1 and traits.reward_for_rank(2) == 0


def test_threshold_requires_clean_comparable_runs():
    baseline = _ts(accuracy=0.7, time_=0.10)
    good = _ts(accuracy=0.9, time_=0.02)
    ev = traits.threshold_evidence(baseline, good, 1.25 / 2.25)
    assert ev["reached"] and ev["proven"] and ev["reasons"] == []

    faster_but_worse = _ts(accuracy=0.6, time_=0.001)
    ev = traits.threshold_evidence(baseline, faster_but_worse, 1.25 / 2.25)
    assert ev["reached"] and not ev["proven"]
    assert any("below the original" in r for r in ev["reasons"])

    crashed = _ts(accuracy=0.0, time_=0.0, re=1)
    ev = traits.threshold_evidence(baseline, crashed, 1.25 / 2.25)
    assert not ev["proven"] and any("runtime" in r for r in ev["reasons"])

    broken_baseline = _ts(accuracy=0.0, time_=60.0, re=1)
    ev = traits.threshold_evidence(broken_baseline, good, 1.25 / 2.25)
    assert ev["reached"] and not ev["proven"]
    assert any("original code failed" in r for r in ev["reasons"])

    ev = traits.threshold_evidence(None, good, 1.25 / 2.25)
    assert not ev["proven"] and any("not been measured" in r for r in ev["reasons"])
