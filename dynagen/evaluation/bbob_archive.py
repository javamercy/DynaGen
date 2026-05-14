from __future__ import annotations

from typing import Any

from dynagen.candidates.candidate import Candidate
from dynagen.evolution.verbal_gradient import metric_float


def build_bbob_archive_profile(candidate: Candidate) -> dict[str, Any]:
    metrics = candidate.metrics or {}
    mean_aocc = metric_float(metrics, "penalized_mean_aocc")
    if mean_aocc is None:
        mean_aocc = metric_float(metrics, "mean_aocc")
    quality_score = _clamp(mean_aocc or 0.0)
    timeout_fraction = metric_float(metrics, "timeout_fraction") or 0.0
    valid_ratio = _ratio(metrics.get("valid_count"), metrics.get("runs"))
    final_error_score = _lower_score(metric_float(metrics, "mean_final_error"), scale=10.0)
    group_scores = _numeric_group(metrics.get("aocc_by_group"))
    worst_group_score = min(group_scores.values()) if group_scores else quality_score
    robustness_score = _clamp(
        0.40 * _clamp(worst_group_score)
        + 0.25 * (1.0 - _clamp(timeout_fraction))
        + 0.20 * final_error_score
        + 0.15 * valid_ratio
    )

    buckets = ["global"]
    bucket_scores = {"global": quality_score}
    for group, value in group_scores.items():
        bucket = f"bbob:group:{_bucket_token(group)}"
        buckets.append(bucket)
        bucket_scores[bucket] = _clamp(value)

    function_scores = _numeric_group(metrics.get("aocc_by_function"))
    for function_id, value in function_scores.items():
        bucket = f"bbob:function:{function_id}"
        buckets.append(bucket)
        bucket_scores[bucket] = _clamp(value)

    if final_error_score >= 0.5:
        buckets.append("bbob:final_error:strong")
        bucket_scores["bbob:final_error:strong"] = final_error_score
    runtime_score = _lower_score(metric_float(metrics, "mean_runtime"), scale=1.0)
    if runtime_score >= 0.5:
        buckets.append("bbob:runtime:fast")
        bucket_scores["bbob:runtime:fast"] = runtime_score
    if timeout_fraction <= 0.0:
        buckets.append("bbob:timeout:robust")
        bucket_scores["bbob:timeout:robust"] = robustness_score

    mechanisms = _bbob_mechanisms(candidate.code)
    for mechanism in mechanisms:
        bucket = f"bbob:mechanism:{mechanism}"
        buckets.append(bucket)
        bucket_scores[bucket] = _clamp(0.5 * quality_score + 0.5 * robustness_score)

    primary_bucket = _best_bucket(group_scores, prefix="bbob:group:") or "global"
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


def _bbob_mechanisms(code: str) -> list[str]:
    text = str(code or "").lower()
    mechanisms: list[str] = []
    if "random" in text or "uniform" in text:
        mechanisms.append("random_search")
    if "population" in text or "elite" in text or "mutation" in text:
        mechanisms.append("evolution_strategy")
    if "differential" in text or "crossover" in text or " cr" in text or " f " in text:
        mechanisms.append("differential_evolution")
    if "cov" in text or "covariance" in text or "mean" in text and "sigma" in text:
        mechanisms.append("cma_like")
    if "restart" in text or "stagnation" in text:
        mechanisms.append("restart")
    if "coordinate" in text or "pattern" in text or "hill" in text:
        mechanisms.append("local_refine")
    return mechanisms[:6]


def _best_bucket(scores: dict[str, float], *, prefix: str) -> str | None:
    if not scores:
        return None
    key, _ = max(scores.items(), key=lambda item: item[1])
    return f"{prefix}{_bucket_token(key)}"


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
        "mean_aocc",
        "penalized_mean_aocc",
        "median_aocc",
        "best_aocc",
        "worst_aocc",
        "mean_final_error",
        "best_final_error",
        "worst_final_error",
        "timeout_fraction",
        "valid_count",
        "runs",
        "mean_runtime",
        "aocc_by_group",
        "aocc_by_function",
        "final_error_by_function",
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
    return str(value).strip().replace(" ", "_").replace("/", "_")


def _float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in {float("inf"), float("-inf")} else None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))

