from __future__ import annotations

from typing import Any

from dynagen.candidates.candidate import Candidate
from dynagen.evolution.verbal_gradient import best_numeric_group, metric_float


def build_vrp_history_profile(candidate: Candidate) -> dict[str, Any]:
    metrics = candidate.metrics or {}
    mean_gap = metric_float(metrics, "penalized_mean_gap")
    if mean_gap is None:
        mean_gap = metric_float(metrics, "mean_gap")
    mean_distance = metric_float(metrics, "mean_max_route_distance")
    distance = metric_float(metrics, "distance")
    score_value = mean_gap if mean_gap is not None else mean_distance if mean_distance is not None else distance
    scale = 25.0 if mean_gap is not None else 10.0
    quality_score = _lower_score(score_value, scale=scale)
    runtime_score = _lower_score(metric_float(metrics, "mean_runtime"), scale=1.0)
    timeout_fraction = metric_float(metrics, "timeout_fraction") or 0.0
    valid_ratio = _ratio(metrics.get("valid_count"), metrics.get("runs"))
    worst_gap = metric_float(metrics, "worst_gap")
    worst_score = _lower_score(worst_gap, scale=50.0) if worst_gap is not None else quality_score
    robustness_score = _clamp(
        0.40 * worst_score
        + 0.25 * (1.0 - _clamp(timeout_fraction))
        + 0.20 * valid_ratio
        + 0.15 * runtime_score
    )

    buckets = ["global"]
    bucket_scores = {"global": quality_score}

    size_scores = _numeric_group(metrics.get("score_by_instance_size"))
    best_size = best_numeric_group(size_scores, higher_is_better=False)
    for size, value in size_scores.items():
        bucket = f"vrp:size:{size}"
        buckets.append(bucket)
        bucket_scores[bucket] = _lower_score(value, scale=scale)

    truck_scores = _numeric_group(metrics.get("score_by_truck_count"))
    for truck_count, value in truck_scores.items():
        bucket = f"vrp:trucks:{truck_count}"
        buckets.append(bucket)
        bucket_scores[bucket] = _lower_score(value, scale=scale)

    source_scores = _numeric_group(metrics.get("score_by_instance_source"))
    for source, value in source_scores.items():
        bucket = f"vrp:source:{_bucket_token(source)}"
        buckets.append(bucket)
        bucket_scores[bucket] = _lower_score(value, scale=scale)

    if runtime_score >= 0.5:
        buckets.append("vrp:runtime:fast")
        bucket_scores["vrp:runtime:fast"] = runtime_score
    if timeout_fraction <= 0.0:
        buckets.append("vrp:runtime:robust")
        bucket_scores["vrp:runtime:robust"] = robustness_score

    mechanisms = _vrp_mechanisms(candidate.code)
    for mechanism in mechanisms:
        bucket = f"vrp:mechanism:{mechanism}"
        buckets.append(bucket)
        bucket_scores[bucket] = _clamp(0.5 * quality_score + 0.5 * robustness_score)

    primary_bucket = f"vrp:size:{best_size[0]}" if best_size else "global"
    if not mechanisms and primary_bucket == "global":
        mechanisms = ["unknown"]

    return {
        "buckets": buckets,
        "primary_bucket": primary_bucket,
        "quality_score": quality_score,
        "robustness_score": robustness_score,
        "bucket_scores": bucket_scores,
        "diversity_features": {
            "mechanisms": mechanisms,
            "strategy": candidate.strategy,
            "status": str(getattr(candidate.status, "value", candidate.status)),
        },
        "metrics_snapshot": _metrics_snapshot(metrics),
    }


def _vrp_mechanisms(code: str) -> list[str]:
    text = str(code or "").lower()
    mechanisms: list[str] = []
    if "nearest" in text or "argmin" in text or "distance" in text:
        mechanisms.append("nearest")
    if "sweep" in text or "angle" in text or "polar" in text:
        mechanisms.append("sweep")
    if "cluster" in text or "kmeans" in text or "sector" in text:
        mechanisms.append("cluster_first")
    if "savings" in text or "clarke" in text or "wright" in text:
        mechanisms.append("savings")
    if "two_opt" in text or "2-opt" in text or "reversed(" in text:
        mechanisms.append("two_opt")
    if "relocate" in text or "exchange" in text or "swap" in text:
        mechanisms.append("route_repair")
    if "restart" in text or "shuffle" in text or "population" in text:
        mechanisms.append("restart_search")
    if "balance" in text or "max_route" in text or "route_load" in text:
        mechanisms.append("fleet_balance")
    return mechanisms[:7]


def _numeric_group(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        number = _float(item)
        if number is not None:
            result[str(key)] = number
    return result


def _metrics_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "gap",
        "distance",
        "mean_gap",
        "penalized_mean_gap",
        "mean_max_route_distance",
        "mean_total_route_distance",
        "median_gap",
        "worst_gap",
        "best_gap",
        "timeout_fraction",
        "valid_count",
        "runs",
        "mean_runtime",
        "score_by_instance_size",
        "score_by_truck_count",
        "score_by_instance_source",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _lower_score(value: float | None, *, scale: float) -> float:
    if value is None:
        return 0.0
    return _clamp(1.0 / (1.0 + max(0.0, float(value)) / max(scale, 1e-9)))


def _ratio(numerator: object, denominator: object) -> float:
    top = _float(numerator)
    bottom = _float(denominator)
    if top is None or bottom is None or bottom <= 0:
        return 0.0
    return _clamp(top / bottom)


def _bucket_token(value: str) -> str:
    token = str(value).strip().replace("\\", "/").split("/")[-1]
    return token.replace(" ", "_")


def _float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in {float("inf"), float("-inf")} else None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
