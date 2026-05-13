import math
import statistics
from collections import defaultdict
from typing import Any


UNSCORED_TIMEOUT_DISTANCE = 1_000_000.0


def compute_dvrp_gap(makespan: float, reference_makespan: float | None) -> float | None:
    if not math.isfinite(makespan) or makespan <= 0:
        raise ValueError("DVRP makespan must be a positive finite value")
    if reference_makespan is None:
        return None
    if not math.isfinite(reference_makespan) or reference_makespan <= 0:
        return None
    return 100.0 * (makespan - reference_makespan) / reference_makespan


def aggregate_dvrp_records(records: list[dict[str, Any]], *, timeout_penalty: float = 0.0) -> dict[str, Any]:
    valid = [record for record in records if record["status"] == "valid"]
    scored = [record for record in records if _has_finite_gap(record)]
    gaps = [float(record["gap"]) for record in scored]
    makespans = [
        float(record["makespan"])
        for record in records
        if record.get("makespan") is not None and math.isfinite(float(record["makespan"]))
    ]
    decisions = [
        float(record["decisions"])
        for record in records
        if record.get("decisions") is not None and math.isfinite(float(record["decisions"]))
    ]
    waits = [
        float(record["waits"])
        for record in records
        if record.get("waits") is not None and math.isfinite(float(record["waits"]))
    ]
    completed_counts = [
        float(record["completed_count"])
        for record in records
        if record.get("completed_count") is not None and math.isfinite(float(record["completed_count"]))
    ]
    runtimes = [float(record.get("runtime_seconds", 0.0)) for record in records]
    timeout_count = sum(1 for record in records if record["status"] == "timeout")
    timeout_fraction = timeout_count / len(records) if records else 0.0
    mean_gap = _mean(gaps)
    penalized_mean_gap = None if mean_gap is None else mean_gap + float(timeout_penalty) * timeout_fraction
    metrics = {
        "runs": len(records),
        "valid_count": len(valid),
        "scored_count": len(scored),
        "timeout_count": timeout_count,
        "invalid_count": sum(1 for record in records if record["status"] == "invalid"),
        "runtime_error_count": sum(1 for record in records if record["status"] == "error"),
        "mean_makespan": _mean(makespans),
        "mean_decisions": _mean(decisions),
        "mean_waits": _mean(waits),
        "mean_completed_count": _mean(completed_counts),
        "mean_gap": mean_gap,
        "timeout_fraction": timeout_fraction,
        "timeout_penalty": float(timeout_penalty),
        "penalized_mean_gap": penalized_mean_gap,
        "timeout_distance": penalized_mean_gap if penalized_mean_gap is not None else (
            UNSCORED_TIMEOUT_DISTANCE if timeout_count else None
        ),
        "median_gap": _median(gaps),
        "worst_gap": max(gaps) if gaps else None,
        "best_gap": min(gaps) if gaps else None,
        "mean_runtime": _mean(runtimes),
        "score_by_instance_size": _group_mean_gap(records, "dimension"),
        "score_by_truck_count": _group_mean_gap(records, "truck_count"),
        "score_by_instance_source": _group_mean_gap(records, "source"),
        "records": records,
    }
    return metrics


def _mean(values: list[float]) -> float | None:
    return None if not values else float(statistics.fmean(values))


def _median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _has_finite_gap(record: dict[str, Any]) -> bool:
    gap = record.get("gap")
    return gap is not None and math.isfinite(gap)


def _group_mean_gap(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    all_keys = {str(record.get(key, "unknown")) for record in records}
    for record in records:
        if _has_finite_gap(record):
            groups[str(record.get(key, "unknown"))].append(float(record["gap"]))
    return {group: _mean(groups[group]) for group in sorted(all_keys)}
