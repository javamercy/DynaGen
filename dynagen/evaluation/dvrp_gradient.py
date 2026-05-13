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


def build_dvrp_static_verbal_gradient(
        candidate: Candidate,
        *,
        parents: list[Candidate],
        generation: int,
) -> dict[str, Any]:
    metrics = candidate.metrics or {}
    status = str(getattr(candidate.status, "value", candidate.status))
    delta = score_delta_vs_best_parent(candidate, parents)
    mean_gap = metric_float(metrics, "mean_gap")
    mean_makespan = metric_float(metrics, "mean_makespan")
    mean_waits = metric_float(metrics, "mean_waits")
    timeout_fraction = metric_float(metrics, "timeout_fraction") or 0.0
    invalid_count = int(metrics.get("invalid_count") or 0)
    runtime_errors = int(metrics.get("runtime_error_count") or 0)
    worst_size = worst_numeric_group(metrics.get("score_by_instance_size"), higher_is_better=False)
    best_size = best_numeric_group(metrics.get("score_by_instance_size"), higher_is_better=False)
    worst_truck = worst_numeric_group(metrics.get("score_by_truck_count"), higher_is_better=False)

    preserve: list[str] = []
    weaknesses: list[str] = []
    avoid: list[str] = []

    if status == "valid":
        preserve.append("cheap online decision rule that returns a valid customer index or None")
        if delta is not None and delta < 0:
            preserve.append("the changed dispatch scoring mechanism that improved over the best parent")
    if mean_gap is not None:
        preserve.append("finite-gap dispatch behavior across the evaluated dynamic instances")
    if mean_makespan is not None:
        preserve.append("depot-return awareness that keeps makespan finite")

    if status == "invalid" or invalid_count:
        weaknesses.append("candidate returned an invalid dispatch decision on at least one simulation")
        avoid.append("index calculations that can point outside available_customers")
    if status == "error" or runtime_errors:
        weaknesses.append("candidate raised a runtime error during online dispatch")
        avoid.append("assumptions about nonempty customer arrays or fixed truck counts")
    if status == "timeout" or timeout_fraction > 0:
        weaknesses.append("candidate timed out during repeated online policy calls")
        avoid.append("expensive pairwise or nested scoring that scales poorly per decision")
    if mean_waits is not None and mean_waits > 0:
        weaknesses.append(f"policy waited {mean_waits:.4g} times on average; unnecessary waiting can increase makespan")
    if worst_size and best_size and worst_size[0] != best_size[0] and worst_size[1] > best_size[1] + 5.0:
        weaknesses.append(f"instance size {worst_size[0]} is the weakest measured group")
    if worst_truck:
        weaknesses.append(f"truck-count bucket {worst_truck[0]} has the weakest measured gap")
    if delta is not None and delta > 0:
        weaknesses.append("child regressed relative to the best selected parent")
        avoid.append("adding terms that improve one route while making fleet finish times less balanced")

    if not weaknesses:
        weaknesses.append("no single measured failure dominates; improve route balance and size robustness")
    if not avoid:
        avoid.append("cosmetic weight changes without a clear dispatch-behavior effect")

    primary_weakness = weaknesses[0]
    next_mutations = {
        "S1": (
            "Explore a different online customer-ranking idea such as spatial partitioning, depot-return pressure, "
            f"or truck competition to target: {primary_weakness}."
        ),
        "S2": (
            "Preserve the current dispatch backbone and retune one or two score terms to directly address: "
            f"{primary_weakness}."
        ),
        "S3": (
            "Use this candidate only for its preserved dispatch strengths, and combine with a parent that covers: "
            f"{primary_weakness}."
        ),
        "default": f"Preserve cheap valid online decisions and address: {primary_weakness}.",
    }
    evidence = {
        "status": status,
        "distance": candidate.score_value,
        "delta_vs_best_parent": delta,
        "mean_gap": mean_gap,
        "median_gap": metrics.get("median_gap"),
        "worst_gap": metrics.get("worst_gap"),
        "best_gap": metrics.get("best_gap"),
        "mean_makespan": mean_makespan,
        "mean_decisions": metrics.get("mean_decisions"),
        "mean_waits": mean_waits,
        "mean_completed_count": metrics.get("mean_completed_count"),
        "timeout_fraction": timeout_fraction,
        "mean_runtime": metrics.get("mean_runtime"),
        "score_by_instance_size": metrics.get("score_by_instance_size"),
        "score_by_truck_count": metrics.get("score_by_truck_count"),
        "error_details": candidate.error_details,
    }
    return base_verbal_gradient(
        problem="dvrp",
        candidate=candidate,
        parents=parents,
        generation=generation,
        summary=_summary(status, delta, mean_gap, mean_makespan, primary_weakness),
        preserve=preserve,
        weaknesses=weaknesses,
        next_mutations=next_mutations,
        avoid=avoid,
        evidence=evidence,
    )


def build_dvrp_llm_verbal_gradient_prompt(
        candidate: Candidate,
        *,
        parents: list[Candidate],
        generation: int,
        static_gradient: dict[str, Any],
) -> list[dict[str, str]]:
    return build_llm_gradient_messages(
        problem="dvrp",
        goal="minimize the time until the last truck returns to the depot using only online dispatch state",
        focus=(
            "customer ranking, waiting behavior, fleet balance, depot-return pressure, spatial clustering, "
            "truck competition, per-call cost, and generalization across customer counts"
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
        mean_makespan: float | None,
        weakness: str,
) -> str:
    score_text = ""
    if mean_gap is not None:
        score_text = f" mean gap {mean_gap:.4g}."
    elif mean_makespan is not None:
        score_text = f" mean makespan {mean_makespan:.4g}."
    if delta is not None:
        direction = "improved over" if delta < 0 else "regressed against" if delta > 0 else "tied"
        return f"{status} DVRP candidate {direction} its best parent by {abs(delta):.4g};{score_text} Main mutation target: {weakness}."
    return f"{status} DVRP candidate;{score_text} Main mutation target: {weakness}."
