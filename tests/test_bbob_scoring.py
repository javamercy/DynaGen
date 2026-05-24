import unittest

from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate
from dynagen.domain.bbob import BBOBInstance
from dynagen.evaluation.bbob_evaluator import BBOBCandidateEvaluator


class BBOBMeanAOCCScoringTests(unittest.TestCase):
    def test_invalid_candidate_uses_mean_aocc_zero_not_fitness(self) -> None:
        evaluator = BBOBCandidateEvaluator(
            [BBOBInstance(function_id=1, instance_id=1, dimension=2)],
            seeds=[100],
            budget=10,
            timeout_seconds=1,
            timeout_penalty=0,
            pool_name="search_instances",
        )
        candidate = Candidate(
            id="cand_1",
            generation=0,
            strategy="initial:1",
            name="invalid",
            thought="",
            code="def not_an_optimizer():\n    pass",
            status=CandidateStatus.PENDING,
        )

        result = evaluator.evaluate_candidate(candidate)

        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.score_name, "mean_aocc")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(candidate.score_name, "mean_aocc")
        self.assertEqual(candidate.score_value, 0.0)
        self.assertIsNone(candidate.fitness)
        self.assertIsNone(candidate.distance)
        self.assertEqual(candidate.metrics["mean_aocc"], 0.0)


if __name__ == "__main__":
    unittest.main()
