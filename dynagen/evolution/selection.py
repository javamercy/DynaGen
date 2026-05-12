import random

from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate

_STATUS_ORDER = {
    CandidateStatus.VALID: 0,
    CandidateStatus.EVALUATED: 0,
    CandidateStatus.TIMEOUT: 1,
    CandidateStatus.INVALID: 2,
    CandidateStatus.ERROR: 3,
    CandidateStatus.PENDING: 4,
}


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

    sorted_candidates = sorted(candidates, key=_sort_key)
    return sorted_candidates[:population_size]


def _rank_biased_probabilities(candidates: list[Candidate]) -> list[float]:
    if not candidates:
        raise ValueError("Cannot select from an empty candidate list")

    sorted_candidates = sorted(candidates, key=_sort_key)
    population_size = len(sorted_candidates)
    weights_by_id = {}

    for rank, candidate in enumerate(sorted_candidates, start=1):
        weights_by_id[candidate.id] = 1.0 / (rank + population_size)

    total = sum(weights_by_id.values())
    return [weights_by_id[candidate.id] / total for candidate in candidates]


def _sort_key(candidate: Candidate) -> tuple[int, float, str]:
    status_rank = _STATUS_ORDER.get(candidate.status, 99)
    score = candidate.score_value if candidate.score_value is not None else float("inf")
    runtime = _metric_float(candidate, "mean_runtime")
    # Prefer newer candidates when the score is tied, then keep ordering stable by ID.
    return status_rank, score, runtime, -int(candidate.generation), candidate.id


def _metric_float(candidate: Candidate, key: str) -> float:
    if not isinstance(candidate.metrics, dict):
        return float("inf")
    value = candidate.metrics.get(key)
    if value is None:
        return float("inf")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return number if number == number else float("inf")
