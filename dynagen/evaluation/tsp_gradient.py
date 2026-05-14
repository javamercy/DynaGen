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


def build_tsp_static_verbal_gradient(
        candidate: Candidate,
        *,
        parents: list[Candidate],
        generation: int,
) -> dict[str, Any]:
    metrics = candidate.metrics or {}
    status = str(getattr(candidate.status, "value", candidate.status))
    delta = score_delta_vs_best_parent(candidate, parents)
    mean_gap = metric_float(metrics, "mean_gap")
    worst_gap = metric_float(metrics, "worst_gap")
    mean_tour_length = metric_float(metrics, "mean_tour_length")
    timeout_fraction = metric_float(metrics, "timeout_fraction") or 0.0
    invalid_tours = int(metrics.get("invalid_tour_count") or 0)
    runtime_errors = int(metrics.get("runtime_error_count") or 0)
    worst_size = worst_numeric_group(metrics.get("score_by_instance_size"), higher_is_better=False)
    best_size = best_numeric_group(metrics.get("score_by_instance_size"), higher_is_better=False)
    worst_source = worst_numeric_group(metrics.get("score_by_instance_source"), higher_is_better=False)

    preserve: list[str] = []
    weaknesses: list[str] = []
    avoid: list[str] = []

    if status == "valid":
        preserve.append("valid tour construction and every return path that preserves a permutation")
        preserve.append("early incumbent reporting with report_best_tour")
        if delta is not None and delta < 0:
            preserve.append("the changed mechanism that improved search distance over the best parent")
    if mean_gap is not None:
        preserve.append("budget-aware improvement behavior that keeps mean gap finite")
    elif mean_tour_length is not None:
        preserve.append("budget-aware improvement behavior that keeps mean tour length finite")

    if status == "invalid" or invalid_tours:
        weaknesses.append("candidate produced an invalid tour on at least one run")
        avoid.append("brittle repair logic that can duplicate, omit, or repeat nodes")
    if status == "error" or runtime_errors:
        weaknesses.append("candidate raised a runtime error under sandbox evaluation")
        avoid.append("untested edge-case assumptions about matrix shape, n <= 2, or helper state")
    if status == "timeout" or timeout_fraction > 0:
        weaknesses.append("candidate timed out or relied on partial timeout scoring")
        avoid.append("unbounded all-pairs neighborhoods and nested loops that scale poorly with n")
    if worst_gap is not None and mean_gap is not None and worst_gap > mean_gap + max(5.0, abs(mean_gap) * 0.4):
        weaknesses.append(f"worst-case gap {worst_gap:.4g} is much weaker than mean gap {mean_gap:.4g}")
    if worst_size and best_size and worst_size[0] != best_size[0] and worst_size[1] > best_size[1] + 5.0:
        weaknesses.append(f"size bucket {worst_size[0]} is the weakest measured group")
    if worst_source and worst_source[1] is not None:
        weaknesses.append(f"source bucket {worst_source[0]} has the weakest measured gap")
    if delta is not None and delta > 0:
        weaknesses.append("child regressed relative to the best selected parent")
        avoid.append("discarding parent mechanisms before isolating which change caused the regression")

    if not weaknesses:
        weaknesses.append("no single failure mode dominates; target generalization and marginal quality gains")
    if not avoid:
        avoid.append("cosmetic changes that only rename variables or nudge constants without changing behavior")

    primary_weakness = weaknesses[0]
    summary = _summary(status, delta, mean_gap, mean_tour_length, primary_weakness)
    next_mutations = {
        "S1": (
            "Explore a materially different construction or neighborhood schedule while preserving validity, "
            f"and target this measured weakness: {primary_weakness}."
        ),
        "S2": (
            "Keep the strongest current structure, make one focused change, and directly address: "
            f"{primary_weakness}."
        ),
        "S3": (
            "Use this candidate only for its preserved mechanisms, combine with another parent that covers: "
            f"{primary_weakness}."
        ),
        "default": f"Preserve valid anytime behavior and address: {primary_weakness}.",
    }
    evidence = {
        "status": status,
        "distance": candidate.score_value,
        "delta_vs_best_parent": delta,
        "mean_tour_length": mean_tour_length,
        "mean_gap": mean_gap,
        "median_gap": metrics.get("median_gap"),
        "worst_gap": worst_gap,
        "best_gap": metrics.get("best_gap"),
        "timeout_fraction": timeout_fraction,
        "mean_runtime": metrics.get("mean_runtime"),
        "score_by_instance_size": metrics.get("score_by_instance_size"),
        "score_by_instance_source": metrics.get("score_by_instance_source"),
        "gap_by_instance_size": metrics.get("gap_by_instance_size"),
        "gap_by_instance_source": metrics.get("gap_by_instance_source"),
        "tour_length_by_instance_size": metrics.get("tour_length_by_instance_size"),
        "tour_length_by_instance_source": metrics.get("tour_length_by_instance_source"),
        "error_details": candidate.error_details,
    }
    return base_verbal_gradient(
        problem="tsp",
        candidate=candidate,
        parents=parents,
        generation=generation,
        summary=summary,
        preserve=preserve,
        weaknesses=weaknesses,
        next_mutations=next_mutations,
        avoid=avoid,
        evidence=evidence,
    )


def build_tsp_llm_verbal_gradient_prompt(
        candidate: Candidate,
        *,
        parents: list[Candidate],
        generation: int,
        static_gradient: dict[str, Any],
) -> list[dict[str, str]]:
    return build_llm_gradient_messages(
        problem="tsp",
        goal="minimize TSP tour length while always returning a valid permutation under a strict budget",
        focus=(
            "construction heuristic, local-search neighborhood, restart/diversification behavior, "
            "budget use, report_best_tour use, and size robustness"
        ),
        candidate=candidate,
        parents=parents,
        generation=generation,
        static_gradient=static_gradient,
    )


def _summary(
        status: str,
        delta: float | None,
        mean_gap: float | None,
        mean_tour_length: float | None,
        weakness: str,
) -> str:
    score_text = ""
    if mean_gap is not None:
        score_text = f" mean gap {mean_gap:.4g}."
    elif mean_tour_length is not None:
        score_text = f" mean tour length {mean_tour_length:.4g}."
    if delta is not None:
        direction = "improved over" if delta < 0 else "regressed against" if delta > 0 else "tied"
        return f"{status} TSP candidate {direction} its best parent by {abs(delta):.4g};{score_text} Main mutation target: {weakness}."
    return f"{status} TSP candidate;{score_text} Main mutation target: {weakness}."
