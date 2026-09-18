
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import memory
from .config import MU_THRESHOLD_DEFAULT

INCLUDE_ERROR_PENALTY_IN_PERFORMANCE = True

MU_THRESHOLD = MU_THRESHOLD_DEFAULT


_MIN_TIME = 1e-6
_MIN_RESOURCE = 1e-6
_MIN_LOSS = 1e-9


@dataclass
class PrimaryTraits:
    training_time: float = 0.0       # TT, seconds
    inference_time: float = 0.0      # TI, seconds
    syntax_errors: int = 0           # SE
    runtime_errors: int = 0          # RE
    logical_errors: int = 0          # EL
    loss: float = 1.0                # L
    model_accuracy: float = 0.0      # Ma
    memory_usage: float = 0.0        # Mu, peak RSS in MB
    compute_cost: float = 0.0        # K, CPU seconds


@dataclass
class SecondaryTraits:
    """The derived traits. `trait` (Tθ) is what agents are ranked by."""

    time: float = 0.0                # T  = TT + TI
    model_performance: float = 0.0   # Mp = Ma / L
    novelty: float = 0.0             # Nθ
    accuracy_term: float = 0.0       # Aθ
    resource: float = 0.0            # R  = K × Mu
    performance: float = 0.0         # P
    trait: float = 0.0               # Tθ = P
    task_performance: float = 0.0    # P* = P with Nθ factored out (used for μ)


@dataclass
class TraitSet:
    """A candidate's full trait profile: what was measured, what was
    derived from it, and the code it describes."""

    primary: PrimaryTraits = field(default_factory=PrimaryTraits)
    secondary: SecondaryTraits = field(default_factory=SecondaryTraits)
    code: str = ""
    stderr: str = ""

    @property
    def trait(self) -> float:
        return self.secondary.trait

    @property
    def performance(self) -> float:
        return self.secondary.performance

    @property
    def novelty(self) -> float:
        return self.secondary.novelty

    @property
    def accuracy(self) -> float:
        return self.primary.model_accuracy

    def has_errors(self) -> bool:
        return (
            self.primary.syntax_errors > 0
            or self.primary.runtime_errors > 0
            or self.primary.logical_errors > 0
        )


def compute_time(training_time: float, inference_time: float) -> float:
    return training_time + inference_time


def compute_model_performance(model_accuracy: float, loss: float) -> float:
    return model_accuracy / max(loss, _MIN_LOSS)


def token_similarity(code_a: str, code_b: str) -> float:
    return memory.cosine_similarity(memory.embed(code_a), memory.embed(code_b))


def compute_novelty(sample_code: str, original_code: str, previous_code: str | None) -> float:
    reference_previous = previous_code if previous_code else original_code
    sim_original = token_similarity(sample_code, original_code)
    sim_previous = token_similarity(sample_code, reference_previous)
    novelty = 1.0 - (sim_original + sim_previous) / 2.0
    return min(1.0, max(0.0, novelty))


def compute_accuracy_term(syntax_errors: int, runtime_errors: int, logical_errors: int) -> float:
    return (2.0 ** (-(syntax_errors + runtime_errors))) * math.exp(-logical_errors)


def compute_resource(compute_cost: float, memory_usage: float) -> float:
    return max(compute_cost * memory_usage, _MIN_RESOURCE)


def compute_performance(
    accuracy_term: float,
    model_accuracy: float,
    novelty: float,
    time: float,
    resource: float,
) -> float:

    numerator = model_accuracy * novelty
    if INCLUDE_ERROR_PENALTY_IN_PERFORMANCE:
        numerator *= accuracy_term
    denominator = max(time, _MIN_TIME) * max(resource, _MIN_RESOURCE)
    return numerator / denominator


def compute_mu(performance_original: float, performance_evolved: float) -> float:
    denominator = performance_original + performance_evolved
    if denominator <= 1e-12:
        return 0.0
    return performance_evolved / denominator


def threshold_evidence(baseline: TraitSet | None, sample: TraitSet, threshold: float) -> dict:
    """Whether a candidate has *proven* it crossed mu, not merely computed a number.

    mu = P2 / (P1 + P2) is only meaningful when both P1 and P2 come from clean,
    comparable runs. Each failed condition is listed in `reasons`; `proven` is
    true only when the list is empty and mu exceeds the threshold.
    """
    reasons: list[str] = []
    p1 = 0.0
    if baseline is None:
        reasons.append("the original code has not been measured")
    else:
        p1 = baseline.secondary.task_performance
        if baseline.primary.syntax_errors or baseline.primary.runtime_errors:
            reasons.append("the original code failed in the sandbox, so P1 is not a valid reference")
        elif p1 <= 0.0:
            reasons.append("the original code's performance P1 is zero")

    p2 = sample.secondary.task_performance
    if sample.primary.syntax_errors:
        reasons.append("the candidate has a syntax error")
    if sample.primary.runtime_errors:
        reasons.append("the candidate failed at runtime")
    if sample.primary.logical_errors:
        reasons.append(f"the candidate has {sample.primary.logical_errors} logical error(s)")
    if baseline is not None and sample.primary.model_accuracy + 1e-9 < baseline.primary.model_accuracy:
        reasons.append(
            f"the candidate's accuracy {sample.primary.model_accuracy:.4f} is below the original's "
            f"{baseline.primary.model_accuracy:.4f}; a faster but worse model does not count"
        )

    mu = compute_mu(p1, p2)
    reached = mu > threshold
    return {
        "mu": mu,
        "threshold": threshold,
        "performance_original": p1,
        "performance_candidate": p2,
        "reached": reached,
        "proven": reached and not reasons,
        "reasons": reasons,
    }


def derive(
    primary: PrimaryTraits,
    sample_code: str,
    original_code: str,
    previous_code: str | None,
    is_reference: bool = False,
) -> SecondaryTraits:
    time = compute_time(primary.training_time, primary.inference_time)
    model_performance = compute_model_performance(primary.model_accuracy, primary.loss)
    novelty = 1.0 if is_reference else compute_novelty(sample_code, original_code, previous_code)
    accuracy_term = compute_accuracy_term(
        primary.syntax_errors, primary.runtime_errors, primary.logical_errors
    )
    resource = compute_resource(primary.compute_cost, primary.memory_usage)
    performance = compute_performance(accuracy_term, primary.model_accuracy, novelty, time, resource)

    task_performance = compute_performance(accuracy_term, primary.model_accuracy, 1.0, time, resource)

    return SecondaryTraits(
        time=time,
        model_performance=model_performance,
        novelty=novelty,
        accuracy_term=accuracy_term,
        resource=resource,
        performance=performance,
        trait=performance,  # Tθ = P
        task_performance=task_performance,
    )


def build(
    primary: PrimaryTraits,
    sample_code: str,
    original_code: str,
    previous_code: str | None = None,
    stderr: str = "",
    is_reference: bool = False,
) -> TraitSet:

    return TraitSet(
        primary=primary,
        secondary=derive(primary, sample_code, original_code, previous_code, is_reference),
        code=sample_code,
        stderr=stderr,
    )



def rank_positions(named_traits: list[tuple[str, float]]) -> list[tuple[str, int]]:

    return [
        (name, 1 + sum(1 for _, other_trait in named_traits if other_trait > trait))
        for name, trait in named_traits
    ]


def reward_for_rank(rank_position: int) -> int:
    return 1 if rank_position == 1 else 0


def format_traits_markdown(label: str, traits: TraitSet) -> str:
    p, s = traits.primary, traits.secondary
    return (
        f"### {label}\n\n"
        f"| Trait | Symbol | Value |\n| --- | --- | --- |\n"
        f"| Training time | TT | {p.training_time:.4f} s |\n"
        f"| Inference time | TI | {p.inference_time:.4f} s |\n"
        f"| Syntax errors | SE | {p.syntax_errors} |\n"
        f"| Runtime errors | RE | {p.runtime_errors} |\n"
        f"| Logical errors | EL | {p.logical_errors} |\n"
        f"| Loss | L | {p.loss:.4f} |\n"
        f"| Model accuracy | Ma | {p.model_accuracy:.4f} |\n"
        f"| Memory usage | Mu | {p.memory_usage:.2f} MB |\n"
        f"| Computational cost | K | {p.compute_cost:.4f} s |\n"
        f"| Time | T | {s.time:.4f} s |\n"
        f"| Model performance | Mp | {s.model_performance:.4f} |\n"
        f"| Novelty | Nθ | {s.novelty:.4f} |\n"
        f"| Accuracy term | Aθ | {s.accuracy_term:.4f} |\n"
        f"| Resource | R | {s.resource:.4f} |\n"
        f"| Task performance (novelty-free) | P* | {s.task_performance:.6g} |\n"
        f"| **Performance** | **P** | **{s.performance:.6g}** |\n"
        f"| **Trait** | **Tθ** | **{s.trait:.6g}** |\n"
    )
