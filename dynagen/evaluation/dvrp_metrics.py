import math
import statistics
from collections import defaultdict
from typing import Any


UNSCORED_TIMEOUT_TTT = 1_000_000.0
UNSCORED_TIMEOUT_GAP = 1_000_000.0


def compute_dvrp_gap(ttt: float, reference_ttt: float | None) -> float | None:
    if not math.isfinite(ttt) or ttt <= 0:
        raise ValueError("DVRP TTT must be a positive finite value")
    if reference_ttt is None:
        return None
    if not math.isfinite(reference_ttt) or reference_ttt <= 0:
        return None
    return 100.0 * (ttt - reference_ttt) / reference_ttt


def aggregate_dvrp_records(records: list[dict[str, Any]], *, timeout_penalty: float = 0.0) -> dict[str, Any]:
    valid = [record for record in records if record["status"] == "valid"]
    scored = [record for record in records if _has_finite_gap(record)]
    gaps = [float(record["gap"]) for record in scored]
    ttts = [ttt for record in records if (ttt := _record_ttt(record)) is not None]
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
    mean_ttt = _mean(ttts)
    penalized_mean_gap = None if mean_gap is None else mean_gap + float(timeout_penalty) * timeout_fraction
    penalized_mean_ttt = None if mean_ttt is None else mean_ttt + float(timeout_penalty) * timeout_fraction
    metrics = {
        "runs": len(records),
        "valid_count": len(valid),
        "scored_count": len(scored),
        "timeout_count": timeout_count,
        "invalid_count": sum(1 for record in records if record["status"] == "invalid"),
        "runtime_error_count": sum(1 for record in records if record["status"] == "error"),
        "mean_ttt": mean_ttt,
        "mean_decisions": _mean(decisions),
        "mean_waits": _mean(waits),
        "mean_completed_count": _mean(completed_counts),
        "mean_gap": mean_gap,
        "timeout_fraction": timeout_fraction,
        "timeout_penalty": float(timeout_penalty),
        "penalized_mean_gap": penalized_mean_gap,
        "penalized_mean_ttt": penalized_mean_ttt,
        "timeout_ttt": penalized_mean_ttt if penalized_mean_ttt is not None else (
            UNSCORED_TIMEOUT_TTT if timeout_count else None
        ),
        "timeout_gap": penalized_mean_gap if penalized_mean_gap is not None else (
            UNSCORED_TIMEOUT_GAP if timeout_count else None
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
        "ttt_by_instance_size": _group_mean_ttt(records, "dimension"),
        "ttt_by_truck_count": _group_mean_ttt(records, "truck_count"),
        "ttt_by_instance_source": _group_mean_ttt(records, "source"),
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


def _record_ttt(record: dict[str, Any]) -> float | None:
    value = record.get("ttt")
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _group_mean_ttt(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    all_keys = {str(record.get(key, "unknown")) for record in records}
    for record in records:
        ttt = _record_ttt(record)
        if ttt is not None:
            groups[str(record.get(key, "unknown"))].append(ttt)
    return {group: _mean(groups[group]) for group in sorted(all_keys)}


def _group_mean_gap(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    all_keys = {str(record.get(key, "unknown")) for record in records}
    for record in records:
        if _has_finite_gap(record):
            groups[str(record.get(key, "unknown"))].append(float(record["gap"]))
    return {group: _mean(groups[group]) for group in sorted(all_keys)}


def _primary_score(record: dict[str, Any]) -> float | None:
    if _has_finite_gap(record):
        return float(record["gap"])
    ttt = record.get("ttt")
    if ttt is not None and math.isfinite(float(ttt)):
        return float(ttt)
    return None


def _group_mean_primary_score(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    all_keys = {str(record.get(key, "unknown")) for record in records}
    for record in records:
        score = _primary_score(record)
        if score is not None:
            groups[str(record.get(key, "unknown"))].append(score)
    return {group: _mean(groups[group]) for group in sorted(all_keys)}
