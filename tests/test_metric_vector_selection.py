import unittest

from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate
from dynagen.evolution.selection import select_survivors


class MetricVectorSelectionTests(unittest.TestCase):
    def test_close_scores_prefer_more_robust_candidate(self) -> None:
        slightly_better_but_fragile = _tsp_candidate(
            "cand_1",
            score=100.0,
            worst_size=160.0,
            timeout_fraction=0.25,
            runtime=4.0,
            code="def solve_tsp(a,b,c):\n    # nearest exhaustive 2-opt\n    return []",
        )
        slightly_worse_but_robust = _tsp_candidate(
            "cand_2",
            score=100.3,
            worst_size=105.0,
            timeout_fraction=0.0,
            runtime=0.5,
            code="def solve_tsp(a,b,c):\n    # candidate insertion restart\n    return []",
        )

        survivors = select_survivors([slightly_better_but_fragile, slightly_worse_but_robust], 1)

        self.assertEqual(survivors[0].id, "cand_2")

    def test_large_score_gap_still_prefers_better_mean_score(self) -> None:
        better_score = _tsp_candidate(
            "cand_1",
            score=100.0,
            worst_size=150.0,
            timeout_fraction=0.1,
            runtime=3.0,
            code="def solve_tsp(a,b,c):\n    return []",
        )
        much_worse_score = _tsp_candidate(
            "cand_2",
            score=115.0,
            worst_size=105.0,
            timeout_fraction=0.0,
            runtime=0.5,
            code="def solve_tsp(a,b,c):\n    # unique candidate list\n    return []",
        )

        survivors = select_survivors([better_score, much_worse_score], 1)

        self.assertEqual(survivors[0].id, "cand_1")

    def test_timeout_candidate_with_materially_better_score_can_survive_for_any_problem(self) -> None:
        for problem in ("tsp", "bbob", "dvrp", "vrp"):
            with self.subTest(problem=problem):
                timeout_score = 0.9 if problem == "bbob" else 100.0
                valid_score = 0.4 if problem == "bbob" else 130.0
                timeout_but_strong = _problem_candidate(
                    "cand_timeout",
                    problem=problem,
                    score=timeout_score,
                    status=CandidateStatus.TIMEOUT,
                )
                valid_but_weak = _problem_candidate(
                    "cand_valid",
                    problem=problem,
                    score=valid_score,
                    status=CandidateStatus.VALID,
                )

                survivors = select_survivors([valid_but_weak, timeout_but_strong], 1)

                self.assertEqual(survivors[0].id, "cand_timeout")

    def test_bbob_higher_mean_aocc_is_better(self) -> None:
        weak = _problem_candidate("cand_weak", problem="bbob", score=0.25, status=CandidateStatus.VALID)
        strong = _problem_candidate("cand_strong", problem="bbob", score=0.85, status=CandidateStatus.VALID)

        survivors = select_survivors([weak, strong], 1)

        self.assertEqual(survivors[0].id, "cand_strong")

    def test_equal_scores_keep_novel_candidate_over_duplicate(self) -> None:
        duplicate_a = _tsp_candidate(
            "cand_1",
            score=100.0,
            worst_size=100.0,
            timeout_fraction=0.0,
            runtime=1.0,
            code="def solve_tsp(a,b,c):\n    # nearest 2-opt\n    return []",
        )
        duplicate_b = _tsp_candidate(
            "cand_2",
            score=100.0,
            worst_size=100.0,
            timeout_fraction=0.0,
            runtime=1.0,
            code="def solve_tsp(a,b,c):\n    # nearest 2-opt\n    return []",
        )
        novel = _tsp_candidate(
            "cand_3",
            score=100.0,
            worst_size=100.0,
            timeout_fraction=0.0,
            runtime=1.0,
            code="def solve_tsp(a,b,c):\n    # insertion restart candidate-list\n    return []",
        )

        survivors = select_survivors([duplicate_a, duplicate_b, novel], 2)

        self.assertIn("cand_3", {candidate.id for candidate in survivors})


def _tsp_candidate(
        candidate_id: str,
        *,
        score: float,
        worst_size: float,
        timeout_fraction: float,
        runtime: float,
        code: str,
) -> Candidate:
    return Candidate(
        id=candidate_id,
        generation=0,
        strategy="S1",
        name="solver",
        thought="",
        code=code,
        distance=score,
        status=CandidateStatus.VALID,
        metrics={
            "problem": "tsp",
            "score_name": "distance",
            "distance": score,
            "runs": 4,
            "valid_count": 4,
            "mean_tour_length": score,
            "timeout_fraction": timeout_fraction,
            "mean_runtime": runtime,
            "score_by_instance_size": {"33": score, "201": worst_size},
            "score_by_instance_source": {"synthetic:tsp_construct:n_instance=11:n_cities=200:seed=11": worst_size},
        },
    )


def _problem_candidate(
        candidate_id: str,
        *,
        problem: str,
        score: float,
        status: CandidateStatus,
) -> Candidate:
    timeout_fraction = 0.5 if status == CandidateStatus.TIMEOUT else 0.0
    valid_count = 2 if status == CandidateStatus.TIMEOUT else 4
    if problem in {"tsp", "dvrp", "vrp"}:
        score_name = "ttt" if problem == "dvrp" else "distance"
        metrics = {
            "problem": problem,
            "score_name": score_name,
            score_name: score,
            "runs": 4,
            "valid_count": valid_count,
            "timeout_fraction": timeout_fraction,
            "mean_runtime": 1.0,
            "score_by_instance_size": {"small": score, "large": score},
            "score_by_instance_source": {"synthetic": score},
        }
        if problem == "tsp":
            metrics.update({"mean_tour_length": score, "mean_gap": score, "worst_gap": score})
        elif problem == "dvrp":
            metrics.update({"mean_ttt": score, "mean_gap": score, "worst_gap": score})
        else:
            metrics.update({"mean_max_route_distance": score, "mean_gap": score, "worst_gap": score})
        return Candidate(
            id=candidate_id,
            generation=0,
            strategy="S1",
            name=f"{problem}_solver",
            thought="",
            code=f"def {problem}_solver():\n    return None",
            distance=score,
            status=status,
            metrics=metrics,
        )

    return Candidate(
        id=candidate_id,
        generation=0,
        strategy="S1",
        name="bbob_optimizer",
        thought="",
        code="class Optimizer:\n    pass",
        status=status,
        metrics={
            "problem": "bbob",
            "score_name": "mean_aocc",
            "mean_aocc": score,
            "penalized_mean_aocc": score,
            "runs": 4,
            "valid_count": valid_count,
            "timeout_fraction": timeout_fraction,
            "mean_runtime": 1.0,
            "mean_final_error": 1.0 - score,
            "worst_final_error": 1.0 - score,
            "aocc_by_group": {"separable": score, "multimodal": score},
        },
    )


if __name__ == "__main__":
    unittest.main()
