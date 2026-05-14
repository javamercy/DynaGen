import hashlib
import random
from dataclasses import dataclass

from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate

_HARD_STATUS_ORDER = {
    CandidateStatus.VALID: 0,
    CandidateStatus.EVALUATED: 0,
    CandidateStatus.TIMEOUT: 0,
    CandidateStatus.INVALID: 1,
    CandidateStatus.ERROR: 1,
    CandidateStatus.PENDING: 1,
}

_STATUS_BADNESS = {
    CandidateStatus.VALID: 0.0,
    CandidateStatus.EVALUATED: 0.0,
    CandidateStatus.TIMEOUT: 0.35,
    CandidateStatus.INVALID: 1.0,
    CandidateStatus.ERROR: 1.25,
    CandidateStatus.PENDING: 1.5,
}


@dataclass(frozen=True)
class _SelectionContext:
    best_score: float
    score_tolerance: float
    worst_group_min: float
    worst_group_max: float
    runtime_min: float
    runtime_max: float
    novelty_by_id: dict[str, float]


def select_parents(
        candidates: list[Candidate],
        count: int,
        rng: random.Random | None) -> list[Candidate]:
    if not candidates:
        raise ValueError("Cannot select from an empty candidate list")

    if count <= 0:
        raise ValueError("Count must be positive")

    if rng is None:
        rng = random.Random()

    pool = [
        c for c in candidates
        if c.status in (CandidateStatus.VALID, CandidateStatus.EVALUATED, CandidateStatus.TIMEOUT)
    ]
    if not pool:
        pool = list(candidates)

    selected: list[Candidate] = []
    for _ in range(min(count, len(pool))):
        probabilities = _rank_biased_probabilities(pool)
        chosen = rng.choices(pool, weights=probabilities, k=1)[0]
        selected.append(chosen)
        pool.remove(chosen)
    return selected


def select_survivors(candidates: list[Candidate], population_size: int) -> list[Candidate]:
    if not candidates:
        raise ValueError("Cannot select from an empty candidate list")

    if population_size <= 0:
        raise ValueError("Population size must be positive")

    sorted_candidates = rank_candidates(candidates)
    return sorted_candidates[:population_size]


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    if not candidates:
        return []
    context = _selection_context(candidates)
    return sorted(candidates, key=lambda candidate: _sort_key(candidate, context))


def _rank_biased_probabilities(candidates: list[Candidate]) -> list[float]:
    if not candidates:
        raise ValueError("Cannot select from an empty candidate list")

    sorted_candidates = rank_candidates(candidates)
    population_size = len(sorted_candidates)
    weights_by_id = {}

    for rank, candidate in enumerate(sorted_candidates, start=1):
        weights_by_id[candidate.id] = 1.0 / (rank + population_size)

    total = sum(weights_by_id.values())
    return [weights_by_id[candidate.id] / total for candidate in candidates]


def _sort_key(candidate: Candidate, context: _SelectionContext | None = None) -> tuple:
    if context is None:
        context = _selection_context([candidate])
    hard_status_rank = _HARD_STATUS_ORDER.get(candidate.status, 1)
    status_badness = _STATUS_BADNESS.get(candidate.status, 1.5)
    score = _finite_or_inf(candidate.score_value)
    score_band = _score_band(score, context)
    worst_group = _normalize(_worst_group_badness(candidate), context.worst_group_min, context.worst_group_max)
    timeout_fraction = _metric_float(candidate, "timeout_fraction", default=0.0)
    validity_badness = _validity_badness(candidate)
    runtime = _normalize(_metric_float(candidate, "mean_runtime"), context.runtime_min, context.runtime_max)
    novelty = context.novelty_by_id.get(candidate.id, 0.0)
    # Mean score remains the main pressure, but close-score candidates are separated by robustness and novelty.
    # Timeouts are not a hard wall: a timeout candidate with a materially better score can survive or parent
    # future mutations, while timeout rate and status still act as robustness penalties inside close score bands.
    return (
        hard_status_rank,
        score_band,
        worst_group,
        timeout_fraction,
        validity_badness,
        status_badness,
        runtime,
        -novelty,
        score,
        -int(candidate.generation),
        candidate.id,
    )


def _selection_context(candidates: list[Candidate]) -> _SelectionContext:
    scores = [_finite_or_inf(candidate.score_value) for candidate in candidates]
    finite_scores = [score for score in scores if _is_finite(score)]
    best_score = min(finite_scores) if finite_scores else float("inf")
    worst_score = max(finite_scores) if finite_scores else best_score
    score_span = max(0.0, worst_score - best_score) if _is_finite(best_score) and _is_finite(worst_score) else 0.0
    score_tolerance = max(abs(best_score) * 0.005 if _is_finite(best_score) else 0.0, score_span * 0.02, 1e-9)
    worst_groups = [_worst_group_badness(candidate) for candidate in candidates]
    runtimes = [_metric_float(candidate, "mean_runtime") for candidate in candidates]
    return _SelectionContext(
        best_score=best_score,
        score_tolerance=score_tolerance,
        worst_group_min=_finite_min(worst_groups),
        worst_group_max=_finite_max(worst_groups),
        runtime_min=_finite_min(runtimes),
        runtime_max=_finite_max(runtimes),
        novelty_by_id=_novelty_scores(candidates),
    )


def _score_band(score: float, context: _SelectionContext) -> int:
    if not _is_finite(score) or not _is_finite(context.best_score):
        return 1_000_000
    return int(max(0.0, score - context.best_score) // context.score_tolerance)


def _worst_group_badness(candidate: Candidate) -> float:
    metrics = candidate.metrics if isinstance(candidate.metrics, dict) else {}
    if _is_bbob_candidate(candidate):
        values = _numeric_values(metrics.get("aocc_by_group")) + _numeric_values(metrics.get("aocc_by_function"))
        if values:
            return 1.0 - min(max(0.0, min(1.0, value)) for value in values)
        worst_aocc = _finite_or_none(metrics.get("worst_aocc"))
        if worst_aocc is not None:
            return 1.0 - max(0.0, min(1.0, worst_aocc))
        final_error = _finite_or_none(metrics.get("worst_final_error"))
        if final_error is not None:
            return final_error
        return _finite_or_inf(candidate.score_value)

    group_values: list[float] = []
    for key in (
            "score_by_instance_size",
            "score_by_instance_source",
            "score_by_truck_count",
            "gap_by_instance_size",
            "gap_by_instance_source",
    ):
        group_values.extend(_numeric_values(metrics.get(key)))
    if group_values:
        return max(group_values)
    for key in ("worst_gap", "penalized_mean_gap", "mean_gap", "mean_tour_length", "mean_makespan"):
        value = _finite_or_none(metrics.get(key))
        if value is not None:
            return value
    return _finite_or_inf(candidate.score_value)


def _validity_badness(candidate: Candidate) -> float:
    metrics = candidate.metrics if isinstance(candidate.metrics, dict) else {}
    runs = _finite_or_none(metrics.get("runs"))
    valid_count = _finite_or_none(metrics.get("valid_count"))
    if runs is None or runs <= 0 or valid_count is None:
        return 0.0 if candidate.status in (CandidateStatus.VALID, CandidateStatus.EVALUATED) else 1.0
    invalid_count = _finite_or_none(metrics.get("invalid_tour_count"))
    if invalid_count is None:
        invalid_count = _finite_or_none(metrics.get("invalid_count")) or 0.0
    error_count = _finite_or_none(metrics.get("runtime_error_count")) or 0.0
    valid_ratio = max(0.0, min(1.0, valid_count / runs))
    failure_ratio = max(0.0, min(1.0, (invalid_count + error_count) / runs))
    return max(0.0, min(1.0, 1.0 - valid_ratio + failure_ratio))


def _novelty_scores(candidates: list[Candidate]) -> dict[str, float]:
    features_by_id = {candidate.id: _novelty_features(candidate) for candidate in candidates}
    scores: dict[str, float] = {}
    for candidate in candidates:
        features = features_by_id[candidate.id]
        distances = [
            _jaccard_distance(features, other_features)
            for other_id, other_features in features_by_id.items()
            if other_id != candidate.id
        ]
        scores[candidate.id] = 0.0 if not distances else sum(distances) / len(distances)
    return scores


def _novelty_features(candidate: Candidate) -> set[str]:
    metrics = candidate.metrics if isinstance(candidate.metrics, dict) else {}
    features = {
        f"strategy:{candidate.strategy}",
        f"status:{candidate.status}",
        f"code_hash:{hashlib.sha256(_normalized_code(candidate.code).encode('utf-8')).hexdigest()}",
    }
    archive = metrics.get("archive")
    if isinstance(archive, dict):
        primary_bucket = archive.get("primary_bucket")
        if primary_bucket:
            features.add(f"archive:{primary_bucket}")
        for bucket in archive.get("buckets") or []:
            features.add(f"bucket:{bucket}")
    text = str(candidate.code or "").lower()
    for marker in (
            "two_opt",
            "2-opt",
            "nearest",
            "insert",
            "restart",
            "shuffle",
            "candidate",
            "population",
            "coordinate",
            "covariance",
            "depot",
            "wait",
            "lookahead",
    ):
        if marker in text:
            features.add(f"code:{marker}")
    return features


def _normalized_code(code: str) -> str:
    return "\n".join(line.strip() for line in str(code or "").splitlines() if line.strip())


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def _metric_float(candidate: Candidate, key: str, *, default: float = float("inf")) -> float:
    if not isinstance(candidate.metrics, dict):
        return default
    value = candidate.metrics.get(key)
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if _is_finite(number) else default


def _numeric_values(value: object) -> list[float]:
    if not isinstance(value, dict):
        return []
    result: list[float] = []
    for item in value.values():
        number = _finite_or_none(item)
        if number is not None:
            result.append(number)
    return result


def _normalize(value: float, minimum: float, maximum: float) -> float:
    if not _is_finite(value):
        return 1.0
    if not _is_finite(minimum) or not _is_finite(maximum) or maximum <= minimum:
        return 0.0
    return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))


def _finite_min(values: list[float]) -> float:
    finite = [value for value in values if _is_finite(value)]
    return min(finite) if finite else float("inf")


def _finite_max(values: list[float]) -> float:
    finite = [value for value in values if _is_finite(value)]
    return max(finite) if finite else float("inf")


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if _is_finite(number) else None


def _finite_or_inf(value: object) -> float:
    number = _finite_or_none(value)
    return number if number is not None else float("inf")


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _is_bbob_candidate(candidate: Candidate) -> bool:
    metrics = candidate.metrics if isinstance(candidate.metrics, dict) else {}
    return metrics.get("problem") == "bbob" or "mean_aocc" in metrics or "aocc_by_group" in metrics
