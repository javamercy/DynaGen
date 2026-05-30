import math
import random
from typing import Any, Callable, Literal

from dynagen.candidates.candidate import Candidate

CommitteeMethod = Literal["greedy_cover", "kmeans"]


def select_committee(
        candidates: list[Candidate],
        *,
        per_instance_scores_fn: Callable[[Candidate], dict[str, float]],
        committee_size: int,
        method: CommitteeMethod = "greedy_cover",
) -> tuple[list[Candidate], dict[str, list[str]]]:
    """
    Select N specialists that collectively cover all instances.

    Methods:
      - "greedy_cover": sequential set cover (fast, deterministic)
      - "kmeans": iterative k-means assignment (better VBS, slower)

    Returns:
        (specialists, instance_assignments) where instance_assignments maps
        candidate_id -> list of instance keys assigned to that specialist.
    """
    if method == "kmeans":
        return _select_committee_kmeans(
            candidates,
            per_instance_scores_fn=per_instance_scores_fn,
            committee_size=committee_size,
        )
    return _select_committee_greedy(
        candidates,
        per_instance_scores_fn=per_instance_scores_fn,
        committee_size=committee_size,
    )


def _select_committee_greedy(
        candidates: list[Candidate],
        *,
        per_instance_scores_fn: Callable[[Candidate], dict[str, float]],
        committee_size: int,
) -> tuple[list[Candidate], dict[str, list[str]]]:
    if not candidates or committee_size <= 0:
        return [], {}

    all_instances = _collect_instance_keys(candidates, per_instance_scores_fn)
    if not all_instances:
        return [], {}

    candidate_scores: dict[str, dict[str, float]] = {}
    for c in candidates:
        candidate_scores[c.id] = per_instance_scores_fn(c)

    best_score_per_instance = _best_per_instance(candidates, candidate_scores, all_instances)

    uncovered = set(all_instances)
    specialists: list[Candidate] = []

    for _ in range(min(committee_size, len(candidates))):
        if not uncovered:
            break
        best_candidate: Candidate | None = None
        best_value = -1.0
        for c in candidates:
            if c in specialists:
                continue
            scores = candidate_scores.get(c.id, {})
            if not scores:
                continue
            uncovered_scores = [scores.get(k, 0.0) for k in uncovered if k in scores]
            if not uncovered_scores:
                continue
            mean_score = sum(uncovered_scores) / len(uncovered_scores)
            if mean_score > best_value:
                best_value = mean_score
                best_candidate = c
        if best_candidate is None:
            break
        specialists.append(best_candidate)

        scores = candidate_scores.get(best_candidate.id, {})
        covered = {
            k for k in uncovered
            if scores.get(k, 0.0) >= 0.7 * best_score_per_instance.get(k, 0.0)
        }
        uncovered -= covered

    assignments: dict[str, list[str]] = {}
    for c in specialists:
        assignments[c.id] = []
    for instance in all_instances:
        best_cid = None
        best_val = -1.0
        for c in specialists:
            val = candidate_scores.get(c.id, {}).get(instance, 0.0)
            if val > best_val:
                best_val = val
                best_cid = c.id
        if best_cid:
            assignments[best_cid].append(instance)

    return specialists, assignments


def _select_committee_kmeans(
        candidates: list[Candidate],
        *,
        per_instance_scores_fn: Callable[[Candidate], dict[str, float]],
        committee_size: int,
        max_iterations: int = 30,
        seed: int = 0,
) -> tuple[list[Candidate], dict[str, list[str]]]:
    """
    k-means style: assign instances to candidates, iteratively reassign.
    Centroids must be existing candidates (not synthetic points).
    """
    if not candidates or committee_size <= 0:
        return [], {}

    candidate_scores: dict[str, dict[str, float]] = {}
    for c in candidates:
        candidate_scores[c.id] = per_instance_scores_fn(c)

    all_instances = _collect_instance_keys(candidates, per_instance_scores_fn)
    if not all_instances:
        return [], {}

    rng = random.Random(seed)

    effective_size = min(committee_size, len(candidates))
    cluster_candidates: dict[int, Candidate] = {}
    for i in range(effective_size):
        cluster_candidates[i] = rng.choice(candidates)

    cluster_assignments: dict[int, list[str]] = {i: [] for i in range(effective_size)}
    for inst in all_instances:
        best_cluster = 0
        best_val = -1.0
        for i in range(effective_size):
            val = candidate_scores.get(cluster_candidates[i].id, {}).get(inst, 0.0)
            if val > best_val:
                best_val = val
                best_cluster = i
        cluster_assignments[best_cluster].append(inst)

    # Safety: ensure no cluster is empty after initial assignment
    for i in range(effective_size):
        if not cluster_assignments[i] and all_instances:
            instance = rng.choice(all_instances)
            cluster_assignments[i].append(instance)
            for j in range(effective_size):
                if j != i and instance in cluster_assignments[j]:
                    cluster_assignments[j].remove(instance)

    for _ in range(max_iterations):
        changed = False

        for i in range(effective_size):
            assigned = cluster_assignments[i]
            if not assigned:
                continue
            used_ids = {cluster_candidates[j].id for j in range(effective_size) if j != i}
            best_c = None
            best_mean = -1.0
            for c in candidates:
                if c.id in used_ids:
                    continue
                scores = candidate_scores.get(c.id, {})
                relevant = [scores.get(k, 0.0) for k in assigned if k in scores]
                if not relevant:
                    continue
                mean_score = sum(relevant) / len(relevant)
                if mean_score > best_mean:
                    best_mean = mean_score
                    best_c = c
            if best_c is not None and best_c.id != cluster_candidates[i].id:
                cluster_candidates[i] = best_c
                changed = True

        for inst in all_instances:
            best_cluster = 0
            best_val = -1.0
            for i in range(effective_size):
                val = candidate_scores.get(cluster_candidates[i].id, {}).get(inst, 0.0)
                if val > best_val:
                    best_val = val
                    best_cluster = i
            if inst not in cluster_assignments.get(best_cluster, []):
                for i in range(effective_size):
                    if inst in cluster_assignments.get(i, []):
                        cluster_assignments[i].remove(inst)
                        changed = True
                        break
                cluster_assignments.setdefault(best_cluster, []).append(inst)

        if not changed:
            break

    specialists = [cluster_candidates[i] for i in range(effective_size)]
    assignments: dict[str, list[str]] = {c.id: cluster_assignments.get(i, []) for i, c in enumerate(specialists)}
    return specialists, assignments


def niche_probabilities(
        specialists: list[Candidate],
        instance_assignments: dict[str, list[str]],
        per_instance_scores_fn: Callable[[Candidate], dict[str, float]],
        *,
        improvement_weight: bool = True,
        improvement_power: float = 0.7,
) -> dict[str, float]:
    """
    Compute per-niche generation probabilities based on room for improvement.

    Returns dict mapping candidate_id -> probability (0..1, normalized).
    """
    if not specialists or not improvement_weight:
        return {
            c.id: 1.0 / len(specialists)
            for c in specialists
        } if specialists else {}

    potentials: dict[str, float] = {}
    for c in specialists:
        assigned = instance_assignments.get(c.id, [])
        if not assigned:
            potentials[c.id] = 0.0
            continue
        scores = per_instance_scores_fn(c)
        total = 0.0
        count = 0
        for instance in assigned:
            s = scores.get(instance, 0.0)
            potential = max(0.0, 1.0 - max(0.0, min(1.0, s)))
            total += potential ** improvement_power
            count += 1
        potentials[c.id] = total / max(1, count) if count else 0.0

    total_potential = sum(potentials.values())
    if total_potential <= 0:
        return {c.id: 1.0 / len(specialists) for c in specialists}

    return {
        cid: max(0.0, prob)
        for cid, prob in potentials.items()
    }


def compute_vbs(
        candidates: list[Candidate],
        per_instance_scores_fn: Callable[[Candidate], dict[str, float]],
) -> dict[str, float]:
    """
    Virtual Best Solver: for each instance, take the best score across all candidates.

    Returns dict mapping instance_key -> best_score across all candidates.
    """
    all_instances = _collect_instance_keys(candidates, per_instance_scores_fn)
    if not all_instances:
        return {}

    vbs: dict[str, float] = {}
    for instance in all_instances:
        best = 0.0
        for c in candidates:
            scores = per_instance_scores_fn(c)
            val = scores.get(instance, 0.0)
            if val > best:
                best = val
        vbs[instance] = best
    return vbs


def assign_instances(
        specialists: list[Candidate],
        all_instances: list[str],
        per_instance_scores_fn: Callable[[Candidate], dict[str, float]],
) -> dict[str, list[str]]:
    """
    Assign each instance to the specialist with the best score on it.
    """
    assignments: dict[str, list[str]] = {c.id: [] for c in specialists}
    for instance in all_instances:
        best_cid = None
        best_val = -1.0
        for c in specialists:
            val = per_instance_scores_fn(c).get(instance, 0.0)
            if val > best_val:
                best_val = val
                best_cid = c.id
        if best_cid:
            assignments[best_cid].append(instance)
    return assignments


def plateau_detect(
        improvement_history: list[float],
        *,
        threshold: float = 0.001,
        patience: int = 3,
) -> bool:
    """
    Detects if improvement has plateaued (for committee_loop mode).

    Returns True if the last `patience` chunks all show improvement below threshold.
    """
    if len(improvement_history) < patience:
        return False
    recent = improvement_history[-patience:]
    return all(abs(v) < threshold for v in recent)


def _collect_instance_keys(
        candidates: list[Candidate],
        per_instance_scores_fn: Callable[[Candidate], dict[str, float]],
) -> list[str]:
    all_keys: set[str] = set()
    for c in candidates:
        scores = per_instance_scores_fn(c)
        all_keys.update(scores.keys())
    return sorted(all_keys)


def _best_per_instance(
        candidates: list[Candidate],
        candidate_scores: dict[str, dict[str, float]],
        all_instances: list[str],
) -> dict[str, float]:
    best: dict[str, float] = {}
    for instance in all_instances:
        max_val = 0.0
        for c in candidates:
            val = candidate_scores.get(c.id, {}).get(instance, 0.0)
            if val > max_val:
                max_val = val
        best[instance] = max_val
    return best
