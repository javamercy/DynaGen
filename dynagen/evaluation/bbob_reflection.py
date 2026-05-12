import json
from typing import Any

from dynagen.candidates.candidate import Candidate


def build_bbob_llm_reflection_prompt(
        candidate: Candidate,
        *,
        parents: list[Candidate],
        generation: int,
) -> list[dict[str, str]]:
    evidence = {
        "generation": generation,
        "candidate": _candidate_snapshot(candidate),
        "parents": [_candidate_snapshot(parent) for parent in parents],
        "comparison": _comparison(candidate, parents),
    }
    user = (
        "Reflect on this BBOB offspring relative to its parent(s). The only goal is lower objective value under a strict budget. "
        "Focus on AOCC, final error, incumbent reporting, restart behavior, step-size or covariance adaptation, and whether the optimizer is overfitting to a subset of functions. "
        "Return 3-6 concise bullets with: what changed, what improved or regressed, what to preserve, and the next change to try. "
        "If there is one parent, compare parent + child; if there are two parents, compare both parents + child. No code.\n\n"
        f"Evidence:\n{json.dumps(evidence, sort_keys=True, separators=(',', ':'))}\n\n"
        f"Code excerpt:\n{candidate.code[:2500]}"
    )
    return [
        {"role": "system", "content": "Produce terse, evidence-based BBOB optimizer reflections that compare parent(s) and child."},
        {"role": "user", "content": user},
    ]


def _candidate_snapshot(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "strategy": candidate.strategy,
        "status": _status_value(candidate),
        "fitness": candidate.score_value,
        "thought": candidate.thought,
        "error_details": candidate.error_details,
        "code": candidate.code[:2500],
        "metrics": _summary_metrics(candidate.metrics or {}),
    }


def _comparison(candidate: Candidate, parents: list[Candidate]) -> dict[str, Any]:
    child_fitness = _as_float(candidate.score_value)
    parent_fitnesses = [value for value in (_as_float(parent.score_value) for parent in parents) if value is not None]
    best_parent = min(parent_fitnesses) if parent_fitnesses else None
    comparison = {
        "child_minus_best_parent_fitness": None,
        "parent_count": len(parents),
    }
    if child_fitness is not None and best_parent is not None:
        comparison["child_minus_best_parent_fitness"] = child_fitness - best_parent
    return comparison


def _summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "runs", "valid_count", "timeout_count", "invalid_count", "runtime_error_count",
        "mean_aocc", "penalized_mean_aocc", "median_aocc", "best_aocc", "worst_aocc",
        "mean_final_error", "best_final_error", "mean_runtime", "aocc_by_group",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _status_value(candidate: Candidate) -> str:
    return str(getattr(candidate.status, "value", candidate.status))


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number else None
