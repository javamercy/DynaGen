import math
import statistics
from collections import defaultdict
from typing import Any


def compute_gap(tour_length: float, optimal_length: float) -> float:
    if not math.isfinite(tour_length) or tour_length <= 0:
        raise ValueError("Tour length must be a positive finite value")

    if not math.isfinite(optimal_length) or optimal_length <= 0:
        raise ValueError("Reference length must be a positive finite value")

    return 100.0 * (tour_length - optimal_length) / optimal_length


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["status"] == "valid"]
    gaps = [float(record["gap"]) for record in valid if record.get("gap") is not None and math.isfinite(record["gap"])]
    lengths = [float(record["tour_length"]) for record in valid if record.get("tour_length") is not None]
    runtimes = [float(record.get("runtime_seconds", 0.0)) for record in records]
    metrics = {
        "runs": len(records),
        "valid_count": len(valid),
        "timeout_count": sum(1 for record in records if record["status"] == "timeout"),
        "invalid_tour_count": sum(1 for record in records if record["status"] == "invalid"),
        "runtime_error_count": sum(1 for record in records if record["status"] == "error"),
        "mean_tour_length": _mean(lengths),
        "mean_gap": _mean(gaps),
        "median_gap": _median(gaps),
        "worst_gap": max(gaps) if gaps else None,
        "best_gap": min(gaps) if gaps else None,
        "mean_runtime": _mean(runtimes),
        "score_by_instance_size": _group_mean_gap(records, "dimension"),
        "score_by_instance_source": _group_mean_gap(records, "source"),
        "records": records,
    }
    return metrics


def _mean(values: list[float]) -> float | None:
    return None if not values else float(statistics.fmean(values))


def _median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _group_mean_gap(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    all_keys = {str(record.get(key, "unknown")) for record in records}
    for record in records:
        if record["status"] == "valid" and record.get("gap") is not None:
            groups[str(record.get(key, "unknown"))].append(float(record["gap"]))
    return {group: _mean(groups[group]) for group in sorted(all_keys)}
