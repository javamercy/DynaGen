import unittest

from dynagen.config import RunConfig
from dynagen.domain.tsp_synthetic import generate_tsp_construct_instances, parse_tsp_construct_spec
from dynagen.domain.tsp_instance import TSPInstance
from dynagen.evaluation.tsp_evaluator import TSPCandidateEvaluator
from dynagen.evaluation.tsp_metrics import aggregate_tsp_records
from dynagen.problems.tsp import TSPProblem, load_tsp_instances


class TSPSyntheticPoolTests(unittest.TestCase):
    def test_construct_spec_parses_baseline_signature(self) -> None:
        spec = "synthetic:tsp_construct:n_instance=20:n_cities=500:seed=2024"

        self.assertEqual(parse_tsp_construct_spec(spec), (20, 500, 2024))

    def test_generator_matches_baseline_shape_and_count(self) -> None:
        instances = generate_tsp_construct_instances()

        self.assertEqual(len(instances), 20)
        self.assertEqual(instances[0].dimension, 500)
        self.assertEqual(instances[0].coordinates.shape, (500, 2))
        self.assertEqual(instances[0].name, "tsp_construct_500_seed2024_000")
        self.assertEqual(instances[-1].name, "tsp_construct_500_seed2024_019")
        self.assertEqual(
            instances[0].metadata["source"],
            "synthetic:tsp_construct:n_instance=20:n_cities=500:seed=2024",
        )

    def test_loader_generates_baseline_pool_in_dynagen(self) -> None:
        instances = load_tsp_instances("synthetic:tsp_construct:n_instance=20:n_cities=500:seed=2024")

        self.assertEqual(len(instances), 20)
        self.assertEqual(instances[0].name, "tsp_construct_500_seed2024_000")
        self.assertEqual(instances[0].dimension, 500)
        self.assertEqual(instances[-1].name, "tsp_construct_500_seed2024_019")
        self.assertEqual(instances[-1].dimension, 500)
        self.assertEqual(
            instances[-1].metadata["source"],
            "synthetic:tsp_construct:n_instance=20:n_cities=500:seed=2024",
        )

    def test_synthetic_group_scores_fall_back_to_tour_length_without_gap(self) -> None:
        metrics = aggregate_tsp_records([
            {
                "status": "valid",
                "tour_length": 10.0,
                "gap": None,
                "dimension": 33,
                "source": "synthetic:tsp_construct:n_instance=11:n_cities=32:seed=11",
                "runtime_seconds": 0.1,
            },
            {
                "status": "valid",
                "tour_length": 14.0,
                "gap": None,
                "dimension": 33,
                "source": "synthetic:tsp_construct:n_instance=23:n_cities=32:seed=23",
                "runtime_seconds": 0.1,
            },
            {
                "status": "valid",
                "tour_length": 40.0,
                "gap": None,
                "dimension": 51,
                "source": "synthetic:tsp_construct:n_instance=11:n_cities=50:seed=11",
                "runtime_seconds": 0.1,
            },
        ])

        self.assertEqual(metrics["score_by_instance_size"], {"33": 12.0, "51": 40.0})
        self.assertEqual(metrics["gap_by_instance_size"], {"33": None, "51": None})
        self.assertEqual(metrics["tour_length_by_instance_size"], {"33": 12.0, "51": 40.0})

    def test_timeout_logs_reported_best_tour_length(self) -> None:
        instance = TSPInstance.from_distance_matrix(
            "square",
            [
                [0, 1, 10, 1],
                [1, 0, 1, 10],
                [10, 1, 0, 1],
                [1, 10, 1, 0],
            ],
            optimal_length=None,
        )
        evaluator = TSPCandidateEvaluator(
            [instance],
            timeout_seconds=0.5,
            timeout_penalty=0,
            pool_name="search_instances",
        )
        code = """
import numpy as np
import time

def solve_tsp(distance_matrix):
    report_best_tour(np.array([0, 1, 2, 3], dtype=int))
    report_best_tour(np.array([0, 2, 1, 3], dtype=int))
    report_best_tour(np.array([0, 0, 1, 2], dtype=int))
    time.sleep(5)
    return np.array([0, 1, 2, 3], dtype=int)
"""

        result = evaluator.evaluate_code(code)

        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.score, 4.0)
        self.assertEqual(result.metrics["distance"], 4.0)
        self.assertEqual(result.metrics["mean_tour_length"], 4.0)
        self.assertEqual(result.metrics["scored_count"], 1)
        self.assertEqual(result.metrics["partial_timeout_count"], 1)
        self.assertEqual(result.metrics["unscored_timeout_count"], 0)
        self.assertEqual(result.metrics["records"][0]["tour_length"], 4.0)

    def test_tsp_test_evaluator_has_no_timeout(self) -> None:
        config = RunConfig.from_dict({
            "run": {"name": "tsp_test", "output_dir": "runs/test", "seed": 1},
            "llm": {
                "provider": "ollama",
                "model": "fake",
                "temperature": 0.1,
            },
            "evolution": {
                "population_size": 1,
                "generations": 0,
                "offspring_per_strategy": 1,
            },
            "evaluation": {
                "timeout_seconds": 7,
                "timeout_penalty": 0,
                "metric": "average_gap",
            },
            "problem": {"type": "tsp"},
            "data": {
                "search_instances": "synthetic:tsp_construct:n_instance=1:n_cities=5:seed=11",
                "test_instances": "synthetic:tsp_construct:n_instance=1:n_cities=5:seed=12",
            },
        })

        problem = TSPProblem()
        search_evaluator = problem.build_evaluator(config, pool_name="search_instances")
        test_evaluator = problem.build_evaluator(config, pool_name="test_instances")

        self.assertEqual(search_evaluator.timeout_seconds, 7.0)
        self.assertIsNone(test_evaluator.timeout_seconds)

    def test_tsp_evaluator_accepts_no_timeout(self) -> None:
        instance = TSPInstance.from_distance_matrix(
            "tiny",
            [
                [0, 1, 2],
                [1, 0, 3],
                [2, 3, 0],
            ],
            optimal_length=None,
        )
        evaluator = TSPCandidateEvaluator(
            [instance],
            timeout_seconds=None,
            timeout_penalty=0,
            pool_name="test_instances",
        )
        code = """
import numpy as np

def solve_tsp(distance_matrix):
    return np.arange(distance_matrix.shape[0], dtype=int)
"""

        result = evaluator.evaluate_code(code)

        self.assertEqual(result.status, "valid")
        self.assertIsNone(result.metrics["timeout_seconds"])
        self.assertIsNone(result.metrics["records"][0]["timeout_limit_seconds"])


if __name__ == "__main__":
    unittest.main()
