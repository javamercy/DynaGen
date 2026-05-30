"""
Post-hoc committee analysis script. Reads saved candidates from a completed
BBOB run, selects N specialists via greedy cover (same algorithm as the engine),
evaluates them on test instances, and computes VBS (Virtual Best Solver).

Run from project root:
  python runs/bbob/20260530_083640_bbob_pop_5_gen_20_only_memory/post_committee.py
"""
import json
import sys
from pathlib import Path

# ---------- Bootstrap: add project root to path ----------
_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_root))

from dynagen.candidates.candidate import Candidate
from dynagen.domain.bbob import create_bbob_instances
from dynagen.evaluation.bbob_evaluator import BBOBCandidateEvaluator
from dynagen.evolution.committee import assign_instances, compute_vbs, select_committee

# ---------- Config ----------
RUN_DIR = Path(__file__).resolve().parent
COMMITTEE_SIZE = 3  # matches config


def main():
    # 1. Load raw config
    raw = json.loads((RUN_DIR / "config.json").read_text())
    problem_cfg = raw["problem"]
    eval_cfg = raw["evaluation"]

    # 2. Build test evaluator directly
    test_instances = create_bbob_instances(
        function_ids=problem_cfg["function_ids"],
        instance_ids=problem_cfg["test_instances"],
        dimensions=problem_cfg["test_dimensions"],
        bounds=problem_cfg["bounds"],
    )
    test_eval = BBOBCandidateEvaluator(
        test_instances,
        seeds=eval_cfg["seeds"],
        budget=eval_cfg["budget"],
        timeout_seconds=eval_cfg["timeout_seconds"],
        timeout_penalty=eval_cfg["timeout_penalty"],
        pool_name="test_instances",
    )

    # 3. Load all candidates from the candidates/ directory
    candidates: list[Candidate] = []
    for cand_path in sorted((RUN_DIR / "candidates").glob("cand_*.json")):
        code_path = cand_path.with_suffix(".py")
        code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
        c = Candidate.from_dict(json.loads(cand_path.read_text()), code=code)
        candidates.append(c)

    print(f"Loaded {len(candidates)} candidates")

    # 4. Per-instance scores (BBOB: per-function AOCC from search evaluation)
    def per_instance_scores(c: Candidate) -> dict[str, float]:
        metrics = c.metrics if isinstance(c.metrics, dict) else {}
        by_function = metrics.get("aocc_by_function")
        if isinstance(by_function, dict):
            return {str(k): float(v) for k, v in by_function.items()}
        return {}

    # 5. Greedy cover committee selection (same algorithm as engine)
    specialists, assignments = select_committee(
        candidates,
        per_instance_scores_fn=per_instance_scores,
        committee_size=COMMITTEE_SIZE,
    )
    print(f"\nSelected {len(specialists)} specialists:")
    for sp in specialists:
        assigned = assignments.get(sp.id, [])
        scores = per_instance_scores(sp)
        assigned_mean = (
            sum(scores.get(k, 0.0) for k in assigned) / len(assigned)
            if assigned else 0.0
        )
        print(f"  {sp.id} ({sp.name}): {len(assigned)} functions, assigned_mean={assigned_mean:.4f}")

    # 6. Evaluate each specialist on test instances
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

    # 7. Re-assign instances based on test scores (best specialist per function)
    all_instances = sorted(set(
        k for sp in specialists
        for k in per_instance_scores(sp).keys()
    ))
    test_assignments = assign_instances(specialists, all_instances, per_instance_scores_fn=per_instance_scores)

    # 8. Compute VBS from test per-function scores
    test_per_function: dict[str, dict[str, float]] = {}
    for sp in specialists:
        tr = test_results.get(sp.id, {})
        metrics = tr.get("metrics", {})
        af = metrics.get("aocc_by_function", {})
        test_per_function[sp.id] = {
            str(k): float(v) for k, v in af.items()
        } if isinstance(af, dict) else {}

    vbs: dict[str, float] = {}
    for func_id in sorted(all_instances):
        best_val = 0.0
        for sp in specialists:
            val = test_per_function.get(sp.id, {}).get(func_id, 0.0)
            if val > best_val:
                best_val = val
        vbs[func_id] = best_val

    vbs_values = [v for v in vbs.values() if isinstance(v, (int, float))]
    vbs_mean = sum(vbs_values) / len(vbs_values) if vbs_values else 0.0

    # 9. Save everything
    output = {
        "config": {
            "committee_size": COMMITTEE_SIZE,
            "output_mode": "committee_specialist",
        },
        "specialists": [
            {
                "candidate_id": sp.id,
                "name": sp.name,
                "generation": sp.generation,
                "strategy": sp.strategy,
                "search_mean_aocc": sp.metrics.get("mean_aocc"),
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
        "vbs": {
            "per_function": vbs,
            "mean": vbs_mean,
        },
        "per_function_aocc": {
            sp.id: test_per_function.get(sp.id, {})
            for sp in specialists
        },
    }

    output_path = RUN_DIR / "committee_results.json"
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved results to {output_path}")
    print(f"\n=== SUMMARY ===")
    print(f"Specialists: {len(specialists)}")
    for sp in specialists:
        tr = test_results.get(sp.id, {})
        print(f"  {sp.id}: test_aocc={tr.get('score', 'N/A'):.6f}")
    print(f"VBS mean: {vbs_mean:.6f}")
    print(f"Done.")


if __name__ == "__main__":
    main()
