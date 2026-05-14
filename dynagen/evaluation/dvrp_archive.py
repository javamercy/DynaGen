from __future__ import annotations

from typing import Any

from dynagen.candidates.candidate import Candidate
from dynagen.evolution.verbal_gradient import best_numeric_group, metric_float


def build_dvrp_archive_profile(candidate: Candidate) -> dict[str, Any]:
    metrics = candidate.metrics or {}
    mean_gap = metric_float(metrics, "penalized_mean_gap")
    if mean_gap is None:
        mean_gap = metric_float(metrics, "mean_gap")
    mean_makespan = metric_float(metrics, "mean_makespan")
    distance = metric_float(metrics, "distance")
    score_value = mean_gap if mean_gap is not None else mean_makespan if mean_makespan is not None else distance
    scale = 100.0 if mean_gap is not None else 1000.0
    quality_score = _lower_score(score_value, scale=scale)
    timeout_fraction = metric_float(metrics, "timeout_fraction") or 0.0
    valid_ratio = _ratio(metrics.get("valid_count"), metrics.get("runs"))
    worst_gap = metric_float(metrics, "worst_gap")
    worst_score = _lower_score(worst_gap, scale=150.0) if worst_gap is not None else quality_score
    completion_score = _completion_score(metrics)
    wait_score = _lower_score(metric_float(metrics, "mean_waits"), scale=10.0)
    robustness_score = _clamp(
        0.30 * worst_score
        + 0.20 * (1.0 - _clamp(timeout_fraction))
        + 0.20 * valid_ratio
        + 0.15 * completion_score
        + 0.15 * wait_score
    )

    buckets = ["global"]
    bucket_scores = {"global": quality_score}

    size_scores = _numeric_group(metrics.get("score_by_instance_size"))
    best_size = best_numeric_group(size_scores, higher_is_better=False)
    for size, value in size_scores.items():
        bucket = f"dvrp:size:{size}"
        buckets.append(bucket)
        bucket_scores[bucket] = _lower_score(value, scale=scale)

    truck_scores = _numeric_group(metrics.get("score_by_truck_count"))
    for truck_count, value in truck_scores.items():
        bucket = f"dvrp:trucks:{truck_count}"
        buckets.append(bucket)
        bucket_scores[bucket] = _lower_score(value, scale=scale)

    source_scores = _numeric_group(metrics.get("score_by_instance_source"))
    for source, value in source_scores.items():
        bucket = f"dvrp:source:{_bucket_token(source)}"
        buckets.append(bucket)
        bucket_scores[bucket] = _lower_score(value, scale=scale)

    if wait_score >= 0.5:
        buckets.append("dvrp:waits:low")
        bucket_scores["dvrp:waits:low"] = wait_score
    if completion_score >= 0.5:
        buckets.append("dvrp:completion:high")
        bucket_scores["dvrp:completion:high"] = completion_score
    runtime_score = _lower_score(metric_float(metrics, "mean_runtime"), scale=1.0)
    if runtime_score >= 0.5:
        buckets.append("dvrp:runtime:fast")
        bucket_scores["dvrp:runtime:fast"] = runtime_score

    mechanisms = _dvrp_mechanisms(candidate.code)
    for mechanism in mechanisms:
        bucket = f"dvrp:mechanism:{mechanism}"
        buckets.append(bucket)
        bucket_scores[bucket] = _clamp(0.5 * quality_score + 0.5 * robustness_score)

    primary_bucket = f"dvrp:size:{best_size[0]}" if best_size else "global"
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


def _dvrp_mechanisms(code: str) -> list[str]:
    text = str(code or "").lower()
    mechanisms: list[str] = []
    if "nearest" in text or "distance" in text or "norm" in text:
        mechanisms.append("nearest_available")
    if "urgency" in text or "due" in text or "slack" in text or "deadline" in text:
        mechanisms.append("urgency")
    if "truck" in text and ("balance" in text or "idle" in text or "load" in text):
        mechanisms.append("fleet_balance")
    if "wait" in text or "return none" in text:
        mechanisms.append("wait_control")
    if "lookahead" in text or "future" in text or "project" in text:
        mechanisms.append("lookahead")
    if "depot" in text or "return" in text:
        mechanisms.append("return_policy")
    return mechanisms[:6]


def _numeric_group(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        number = _float(item)
        if number is not None:
            result[str(key)] = number
    return result


def _completion_score(metrics: dict[str, Any]) -> float:
    completed = metric_float(metrics, "mean_completed_count")
    if completed is None:
        return _ratio(metrics.get("valid_count"), metrics.get("runs"))
    sizes = _numeric_group(metrics.get("score_by_instance_size"))
    max_size = max((_float(size) or 0.0 for size in sizes), default=0.0)
    if max_size <= 0:
        return _clamp(completed / max(completed, 1.0))
    return _clamp(completed / max_size)


def _metrics_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "distance",
        "mean_gap",
        "penalized_mean_gap",
        "median_gap",
        "worst_gap",
        "best_gap",
        "mean_makespan",
        "mean_decisions",
        "mean_waits",
        "mean_completed_count",
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

