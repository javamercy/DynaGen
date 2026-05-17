import math
import statistics
from collections import defaultdict
from typing import Any

UNSCORED_TIMEOUT_DISTANCE = 1_000_000.0


def compute_vrp_gap(distance: float, reference_distance: float | None) -> float | None:
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("VRP route distance must be a positive finite value")
    if reference_distance is None:
        return None
    if not math.isfinite(reference_distance) or reference_distance <= 0:
        return None
    return 100.0 * (distance - reference_distance) / reference_distance


def aggregate_vrp_records(records: list[dict[str, Any]], *, timeout_penalty: float = 0.0) -> dict[str, Any]:
    valid = [record for record in records if record["status"] == "valid"]
    scored = [record for record in records if _has_finite_gap(record)]
    gaps = [float(record["gap"]) for record in scored]
    max_distances = [
        float(record["max_route_distance"])
        for record in records
        if record.get("max_route_distance") is not None and math.isfinite(float(record["max_route_distance"]))
    ]
    total_distances = [
        float(record["total_route_distance"])
        for record in records
        if record.get("total_route_distance") is not None and math.isfinite(float(record["total_route_distance"]))
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
        "partial_timeout_count": sum(1 for record in scored if record["status"] == "timeout"),
        "unscored_timeout_count": sum(1 for record in records if record["status"] == "timeout" and not _has_finite_gap(record)),
        "invalid_route_count": sum(1 for record in records if record["status"] == "invalid"),
        "runtime_error_count": sum(1 for record in records if record["status"] == "error"),
        "mean_max_route_distance": _mean(max_distances),
        "mean_total_route_distance": _mean(total_distances),
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
        "score_by_instance_size": _group_mean_primary_score(records, "dimension"),
        "score_by_truck_count": _group_mean_primary_score(records, "truck_count"),
        "score_by_instance_source": _group_mean_primary_score(records, "source"),
        "gap_by_instance_size": _group_mean_gap(records, "dimension"),
        "gap_by_truck_count": _group_mean_gap(records, "truck_count"),
        "gap_by_instance_source": _group_mean_gap(records, "source"),
        "max_route_distance_by_instance_size": _group_mean_number(records, "dimension", "max_route_distance"),
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


def _group_mean_primary_score(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    all_keys = {str(record.get(key, "unknown")) for record in records}
    for record in records:
        score = _primary_score(record)
        if score is not None:
            groups[str(record.get(key, "unknown"))].append(score)
    return {group: _mean(groups[group]) for group in sorted(all_keys)}


def _group_mean_gap(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    all_keys = {str(record.get(key, "unknown")) for record in records}
    for record in records:
        if _has_finite_gap(record):
            groups[str(record.get(key, "unknown"))].append(float(record["gap"]))
    return {group: _mean(groups[group]) for group in sorted(all_keys)}


def _group_mean_number(records: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    all_keys = {str(record.get(group_key, "unknown")) for record in records}
    for record in records:
        value = record.get(value_key)
        if value is not None and math.isfinite(float(value)):
            groups[str(record.get(group_key, "unknown"))].append(float(value))
    return {group: _mean(groups[group]) for group in sorted(all_keys)}


def _primary_score(record: dict[str, Any]) -> float | None:
    if _has_finite_gap(record):
        return float(record["gap"])
    distance = record.get("max_route_distance")
    if distance is not None and math.isfinite(float(distance)):
        return float(distance)
    return None
