"""
Same as post_committee.py but uses k-means style iterative assignment
instead of greedy cover for committee selection.
"""
import json
import random
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_root))

from dynagen.candidates.candidate import Candidate
from dynagen.domain.bbob import create_bbob_instances
from dynagen.evaluation.bbob_evaluator import BBOBCandidateEvaluator
from dynagen.evolution.committee import assign_instances

RUN_DIR = Path(__file__).resolve().parent
COMMITTEE_SIZE = 3
KMEANS_ITERATIONS = 300   # > enough for convergence


def kmeans_select_committee(candidates, per_instance_scores_fn, committee_size, *, seed=0):
    """
    k-means style: assign functions to candidates, iteratively reassign.
    Centroids must be existing candidates (not synthetic points).
    """
    candidate_scores = {c.id: per_instance_scores_fn(c) for c in candidates}
    all_instances = sorted(set(k for scores in candidate_scores.values() for k in scores))
    if not all_instances:
        return [], {}

    rng = random.Random(seed)

    # Init: random assignment of instances to N random candidates
    cluster_candidates: dict[int, Candidate] = {}
    for i in range(committee_size):
        cluster_candidates[i] = rng.choice(candidates)

    cluster_assignments: dict[int, list[str]] = {i: [] for i in range(committee_size)}
    for inst in all_instances:
        # assign to nearest centroid (best-scoring candidate for this instance)
        best_cluster = 0
        best_val = -1.0
        for i in range(committee_size):
            val = candidate_scores.get(cluster_candidates[i].id, {}).get(inst, 0.0)
            if val > best_val:
                best_val = val
                best_cluster = i
        cluster_assignments[best_cluster].append(inst)

    # Iterate
    for iteration in range(KMEANS_ITERATIONS):
        changed = False

        # Step A: recompute centroids (pick best candidate for each cluster's assigned instances)
        for i in range(committee_size):
            assigned = cluster_assignments[i]
            if not assigned:
                continue
            best_c = None
            best_mean = -1.0
            for c in candidates:
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

        # Step B: reassign instances to best centroid
        for inst in all_instances:
            best_cluster = 0
            best_val = -1.0
            for i in range(committee_size):
                val = candidate_scores.get(cluster_candidates[i].id, {}).get(inst, 0.0)
                if val > best_val:
                    best_val = val
                    best_cluster = i
            if inst not in cluster_assignments.get(best_cluster, []):
                # Remove from old cluster
                for i in range(committee_size):
                    if inst in cluster_assignments.get(i, []):
                        cluster_assignments[i].remove(inst)
                        changed = True
                        break
                cluster_assignments.setdefault(best_cluster, []).append(inst)

        if not changed:
            break

    # Build result
    specialists = [cluster_candidates[i] for i in range(committee_size)]
    assignments = {c.id: cluster_assignments.get(i, []) for i, c in enumerate(specialists)}
    return specialists, assignments


def main():
    raw = json.loads((RUN_DIR / "config.json").read_text())
    problem_cfg = raw["problem"]
    eval_cfg = raw["evaluation"]

    test_instances = create_bbob_instances(
        function_ids=problem_cfg["function_ids"],
        instance_ids=problem_cfg["test_instances"],
        dimensions=problem_cfg["test_dimensions"],
        bounds=problem_cfg["bounds"],
    )
    test_eval = BBOBCandidateEvaluator(
        test_instances,
        seeds=eval_cfg["seeds"],
        budget=eval_cfg["budget"]*10,
        timeout_seconds=eval_cfg["timeout_seconds"],
        timeout_penalty=eval_cfg["timeout_penalty"],
        pool_name="test_instances",
    )

    candidates: list[Candidate] = []
    for cand_path in sorted((RUN_DIR / "candidates").glob("cand_*.json")):
        code_path = cand_path.with_suffix(".py")
        code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
        c = Candidate.from_dict(json.loads(cand_path.read_text()), code=code)
        candidates.append(c)

    print(f"Loaded {len(candidates)} candidates")

    def per_instance_scores(c: Candidate) -> dict[str, float]:
        metrics = c.metrics if isinstance(c.metrics, dict) else {}
        by_function = metrics.get("aocc_by_function")
        if isinstance(by_function, dict):
            return {str(k): float(v) for k, v in by_function.items()}
        return {}

    # --- K-MEANS committee selection ---
    specialists, assignments = kmeans_select_committee(
        candidates,
        per_instance_scores_fn=per_instance_scores,
        committee_size=COMMITTEE_SIZE,
    )
    print(f"\nSelected {len(specialists)} specialists (k-means):")
    for sp in specialists:
        assigned = assignments.get(sp.id, [])
        scores = per_instance_scores(sp)
        assigned_mean = (
            sum(scores.get(k, 0.0) for k in assigned) / len(assigned)
            if assigned else 0.0
        )
        print(f"  {sp.id} ({sp.name}): {len(assigned)} functions, assigned_mean={assigned_mean:.4f}, gen={sp.generation}, strategy={sp.strategy}")

    # Evaluate each specialist on test instances
    print("\nEvaluating specialists on test instances...")
    test_results: dict[str, dict] = {}
    for sp in specialists:
        print(f"  Testing {sp.id} ({sp.name}) ...")
        result = test_eval.evaluate_code(sp.code)
        test_results[sp.id] = {
            "candidate_id": sp.id,
            "name": sp.name,
            "status": result.status,
            "score": result.score,
            "score_name": result.score_name,
            "metrics": result.metrics,
        }
        print(f"    status={result.status}  mean_aocc={result.score:.6f}")

    all_instances = sorted(set(k for sp in specialists for k in per_instance_scores(sp).keys()))
    test_assignments = assign_instances(specialists, all_instances, per_instance_scores_fn=per_instance_scores)

    test_per_function: dict[str, dict[str, float]] = {}
    for sp in specialists:
        tr = test_results.get(sp.id, {})
        af = (tr.get("metrics") or {}).get("aocc_by_function", {})
        test_per_function[sp.id] = {str(k): float(v) for k, v in af.items()} if isinstance(af, dict) else {}

    vbs: dict[str, float] = {}
    for func_id in sorted(all_instances):
        vbs[func_id] = max((test_per_function.get(sp.id, {}).get(func_id, 0.0)) for sp in specialists)

    vbs_values = [v for v in vbs.values() if isinstance(v, (int, float))]
    vbs_mean = sum(vbs_values) / len(vbs_values) if vbs_values else 0.0

    output = {
        "method": "kmeans",
        "config": {"committee_size": COMMITTEE_SIZE, "kmeans_iterations": KMEANS_ITERATIONS},
        "specialists": [
            {
                "candidate_id": sp.id, "name": sp.name, "generation": sp.generation,
                "strategy": sp.strategy, "search_mean_aocc": sp.metrics.get("mean_aocc"),
                "assigned_functions": assignments.get(sp.id, []),
                "test_assigned_functions": test_assignments.get(sp.id, []),
                "test_status": test_results.get(sp.id, {}).get("status"),
                "test_mean_aocc": test_results.get(sp.id, {}).get("score"),
            }
            for sp in specialists
        ],
        "assignments": assignments,
        "test_assignments": test_assignments,
        "test_results": test_results,
        "vbs": {"per_function": vbs, "mean": vbs_mean},
        "per_function_aocc": {sp.id: test_per_function.get(sp.id, {}) for sp in specialists},
    }

    output_path = RUN_DIR / "committee_results_kmeans.json"
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {output_path}")
    print(f"\n=== SUMMARY (k-means) ===")
    for sp in specialists:
        tr = test_results.get(sp.id, {})
        print(f"  {sp.id}: test_aocc={tr.get('score', 'N/A'):.6f}")
    print(f"VBS mean: {vbs_mean:.6f}")


if __name__ == "__main__":
    main()
