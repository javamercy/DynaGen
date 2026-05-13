from typing import Any

from dynagen.candidates.candidate import Candidate
from dynagen.evolution.verbal_gradient import (
    base_verbal_gradient,
    best_numeric_group,
    build_llm_gradient_messages,
    metric_float,
    score_delta_vs_best_parent,
    worst_numeric_group,
)


def build_bbob_static_verbal_gradient(
        candidate: Candidate,
        *,
        parents: list[Candidate],
        generation: int,
) -> dict[str, Any]:
    metrics = candidate.metrics or {}
    status = str(getattr(candidate.status, "value", candidate.status))
    delta = score_delta_vs_best_parent(candidate, parents)
    mean_aocc = metric_float(metrics, "mean_aocc")
    mean_final_error = metric_float(metrics, "mean_final_error")
    timeout_fraction = metric_float(metrics, "timeout_fraction") or 0.0
    invalid_count = int(metrics.get("invalid_count") or 0)
    runtime_errors = int(metrics.get("runtime_error_count") or 0)
    weak_group = worst_numeric_group(metrics.get("aocc_by_group"), higher_is_better=True)
    strong_group = best_numeric_group(metrics.get("aocc_by_group"), higher_is_better=True)
    weak_function = worst_numeric_group(metrics.get("aocc_by_function"), higher_is_better=True)

    preserve: list[str] = []
    weaknesses: list[str] = []
    avoid: list[str] = []

    if status == "valid":
        preserve.append("strict budget accounting and feasible incumbent reporting")
        if delta is not None and delta < 0:
            preserve.append("the changed mechanism that improved fitness over the best parent")
    if mean_aocc is not None:
        preserve.append("any search phase that contributes to finite AOCC progress")
    if strong_group:
        preserve.append(f"mechanisms that work on {strong_group[0]} functions")

    if status == "invalid" or invalid_count:
        weaknesses.append("candidate violated the optimizer contract or returned invalid output")
        avoid.append("changing Optimizer signatures, bound handling, or return shape")
    if status == "error" or runtime_errors:
        weaknesses.append("candidate raised a runtime error during objective evaluation")
        avoid.append("unguarded assumptions about dimension, bounds, or empty populations")
    if status == "timeout" or timeout_fraction > 0:
        weaknesses.append("candidate timed out or used expensive loops under the evaluation budget")
        avoid.append("inner loops that do not check remaining budget before evaluating")
    if weak_group:
        weaknesses.append(f"weakest BBOB group is {weak_group[0]} with AOCC {weak_group[1]:.4g}")
    if weak_function:
        weaknesses.append(f"weakest function bucket is {weak_function[0]}")
    if mean_final_error is not None and mean_final_error > 1e-6:
        weaknesses.append(f"mean final error remains {mean_final_error:.4g}, suggesting insufficient local refinement")
    if delta is not None and delta > 0:
        weaknesses.append("child regressed relative to the best selected parent")
        avoid.append("stacking multiple optimizer families without resolving budget conflicts")

    if not weaknesses:
        weaknesses.append("no single measured group dominates; improve robustness across BBOB groups")
    if not avoid:
        avoid.append("cosmetic parameter changes without a mechanism-level performance rationale")

    primary_weakness = weaknesses[0]
    next_mutations = {
        "S1": (
            "Explore a different optimizer family or restart/diversification dynamic that directly targets: "
            f"{primary_weakness}."
        ),
        "S2": (
            "Preserve the current backbone and make one focused adaptation to step-size, covariance, coordinate search, "
            f"or restart logic for: {primary_weakness}."
        ),
        "S3": (
            "Use this parent only for its preserved strengths, and combine it with a parent covering: "
            f"{primary_weakness}."
        ),
        "default": f"Preserve budget correctness and address: {primary_weakness}.",
    }
    evidence = {
        "status": status,
        "fitness": candidate.score_value,
        "delta_vs_best_parent": delta,
        "mean_aocc": mean_aocc,
        "median_aocc": metrics.get("median_aocc"),
        "best_aocc": metrics.get("best_aocc"),
        "worst_aocc": metrics.get("worst_aocc"),
        "mean_final_error": mean_final_error,
        "timeout_fraction": timeout_fraction,
        "mean_runtime": metrics.get("mean_runtime"),
        "mean_evaluations": metrics.get("mean_evaluations"),
        "aocc_by_group": metrics.get("aocc_by_group"),
        "aocc_by_function": metrics.get("aocc_by_function"),
        "error_details": candidate.error_details,
    }
    return base_verbal_gradient(
        problem="bbob",
        candidate=candidate,
        parents=parents,
        generation=generation,
        summary=_summary(status, delta, mean_aocc, primary_weakness),
        preserve=preserve,
        weaknesses=weaknesses,
        next_mutations=next_mutations,
        avoid=avoid,
        evidence=evidence,
    )


def build_bbob_llm_verbal_gradient_prompt(
        candidate: Candidate,
        *,
        parents: list[Candidate],
        generation: int,
        static_gradient: dict[str, Any],
) -> list[dict[str, str]]:
    return build_llm_gradient_messages(
        problem="bbob",
        goal="maximize AOCC, equivalently minimize DynaGen fitness, under strict objective-evaluation budgets",
        focus=(
            "step-size control, covariance or coordinate adaptation, restart behavior, multimodal diversification, "
            "local refinement, bound handling, and budget accounting"
        ),
        candidate=candidate,
        parents=parents,
        generation=generation,
        static_gradient=static_gradient,
    )


def _summary(status: str, delta: float | None, mean_aocc: float | None, weakness: str) -> str:
    score_text = "" if mean_aocc is None else f" mean AOCC {mean_aocc:.4g}."
    if delta is not None:
        direction = "improved over" if delta < 0 else "regressed against" if delta > 0 else "tied"
        return f"{status} BBOB candidate {direction} its best parent by {abs(delta):.4g};{score_text} Main mutation target: {weakness}."
    return f"{status} BBOB candidate;{score_text} Main mutation target: {weakness}."
