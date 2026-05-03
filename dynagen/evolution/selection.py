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
    fitness = candidate.fitness if candidate.fitness is not None else float("inf")
    return status_rank, fitness, candidate.id

# DONE
