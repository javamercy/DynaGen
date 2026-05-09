import math
import statistics
from collections import defaultdict
from typing import Any


def compute_aocc(
        best_history: list[float],
        *,
        optimum: float,
        budget: int,
        lower_bound: float = 1e-8,
        upper_bound: float = 1e2,
) -> float:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if lower_bound <= 0 or upper_bound <= lower_bound:
        raise ValueError("AOCC bounds must satisfy 0 < lower < upper")
    if not best_history:
        return 0.0

    log_lower = math.log10(lower_bound)
    log_upper = math.log10(upper_bound)
    span = log_upper - log_lower
    values = list(best_history[:budget])
    if len(values) < budget:
        values.extend([values[-1]] * (budget - len(values)))

    normalized: list[float] = []
    for value in values:
        precision = max(float(value) - float(optimum), lower_bound)
        log_precision = math.log10(min(max(precision, lower_bound), upper_bound))
        normalized.append(1.0 - (log_precision - log_lower) / span)
    return float(statistics.fmean(normalized))


def aggregate_bbob_records(records: list[dict[str, Any]], *, timeout_penalty: float = 0.0) -> dict[str, Any]:
    valid = [record for record in records if record["status"] == "valid"]
    scored = [record for record in records if _finite(record.get("aocc"))]
    aoccs = [float(record["aocc"]) for record in scored]
    final_errors = [float(record["final_error"]) for record in scored if _finite(record.get("final_error"))]
    evaluations = [float(record.get("evaluations", 0.0)) for record in records]
    runtimes = [float(record.get("runtime_seconds", 0.0)) for record in records]
    timeout_count = sum(1 for record in records if record["status"] == "timeout")
    timeout_fraction = timeout_count / len(records) if records else 0.0
    mean_aocc = _mean(aoccs)
    penalized_mean_aocc = None if mean_aocc is None else max(0.0, mean_aocc - float(timeout_penalty) * timeout_fraction)
    unscored_timeout_count = sum(1 for record in records if record["status"] == "timeout" and not _finite(record.get("aocc")))
    return {
        "problem": "bbob",
        "runs": len(records),
        "valid_count": len(valid),
        "scored_count": len(scored),
        "timeout_count": timeout_count,
        "partial_timeout_count": sum(1 for record in scored if record["status"] == "timeout"),
        "unscored_timeout_count": unscored_timeout_count,
        "invalid_count": sum(1 for record in records if record["status"] == "invalid"),
        "runtime_error_count": sum(1 for record in records if record["status"] == "error"),
        "mean_aocc": mean_aocc,
        "penalized_mean_aocc": penalized_mean_aocc,
        "timeout_fitness": 1.0 - penalized_mean_aocc if penalized_mean_aocc is not None else (
            1.0 if timeout_count else None
        ),
        "median_aocc": _median(aoccs),
        "best_aocc": max(aoccs) if aoccs else None,
        "worst_aocc": min(aoccs) if aoccs else None,
        "mean_final_error": _mean(final_errors),
        "median_final_error": _median(final_errors),
        "best_final_error": min(final_errors) if final_errors else None,
        "worst_final_error": max(final_errors) if final_errors else None,
        "mean_evaluations": _mean(evaluations),
        "mean_runtime": _mean(runtimes),
        "timeout_fraction": timeout_fraction,
        "timeout_penalty": float(timeout_penalty),
        "aocc_by_function": _group_mean(records, "function_id", "aocc"),
        "aocc_by_group": _group_mean(records, "group", "aocc"),
        "final_error_by_function": _group_mean(records, "function_id", "final_error"),
        "records": records,
    }


def _mean(values: list[float]) -> float | None:
    return None if not values else float(statistics.fmean(values))


def _median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _group_mean(records: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    all_keys = {str(record.get(group_key, "unknown")) for record in records}
    for record in records:
        value = record.get(value_key)
        if _finite(value):
            groups[str(record.get(group_key, "unknown"))].append(float(value))
    return {group: _mean(groups[group]) for group in sorted(all_keys)}
