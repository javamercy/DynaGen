import unittest

from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate
from dynagen.candidates.validation import validate_vrp_generated_code
from dynagen.config import RunConfig
from dynagen.domain.vrp import VRPSolutionError, evaluate_vrp_routes, load_vrp_instances
from dynagen.evaluation.vrp_evaluator import VRPCandidateEvaluator
from dynagen.evaluation.vrp_metrics import aggregate_vrp_records, compute_vrp_gap
from dynagen.problems.registry import problem_for_config
from dynagen.problems.vrp import VRPProblem


class VRPFlowTests(unittest.TestCase):
    def test_loads_truck_sim_vrp_pickles_from_data_dir(self) -> None:
        train = load_vrp_instances(
            "data/vrp/train/instances.pkl",
            pool_name="search_instances",
            search_limit=2,
        )
        test = load_vrp_instances(
            "data/vrp/test",
            pool_name="test_instances",
            test_sizes=[10],
            test_limit_per_size=2,
        )

        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)
        self.assertEqual(train[0].distance_matrix.shape, (train[0].dimension, train[0].dimension))
        self.assertGreater(train[0].truck_count, 0)
        self.assertIsNotNone(train[0].reference_distance)

    def test_vrp_route_validation_scores_reference_routes(self) -> None:
        instance = load_vrp_instances(
            "data/vrp/test",
            pool_name="test_instances",
            test_sizes=[10],
            test_limit_per_size=1,
        )[0]

        result = evaluate_vrp_routes(instance, instance.reference["routes"])
        gap = compute_vrp_gap(result.max_route_distance, instance.reference_distance)

        self.assertEqual(result.visited_count, instance.customer_count)
        self.assertEqual(len(result.routes), instance.truck_count)
        self.assertIsNotNone(gap)
        self.assertLess(abs(gap or 0.0), 0.1)

    def test_vrp_route_validation_rejects_incomplete_routes(self) -> None:
        instance = load_vrp_instances(
            "data/vrp/test",
            pool_name="test_instances",
            test_sizes=[10],
            test_limit_per_size=1,
        )[0]

        with self.assertRaises(VRPSolutionError):
            evaluate_vrp_routes(instance, [[0, 0] for _ in range(instance.truck_count)])

    def test_vrp_config_and_registry_wiring(self) -> None:
        config = RunConfig.from_dict({
            "run": {"name": "vrp_test", "output_dir": "runs/test", "seed": 1},
            "llm": {
                "provider": "ollama",
                "model": "fake",
                "temperature": 0.1,
            },
            "problem": {
                "type": "vrp",
                "vrp_search_limit": 2,
                "vrp_test_sizes": [10],
                "vrp_test_limit_per_size": 2,
            },
            "evolution": {
                "population_size": 1,
                "generations": 0,
                "offspring_per_strategy": 1,
                "strategies": ["e1_radical_exploration"],
            },
            "evaluation": {
                "budget": 10,
                "timeout_seconds": 1,
                "seeds": [1],
                "metric": "mean_gap",
            },
            "data": {
                "search_instances": "data/vrp/train/instances.pkl",
                "test_instances": "data/vrp/test",
            },
        })

        self.assertEqual(config.problem.type, "vrp")
        self.assertEqual(config.problem.vrp_search_limit, 2)
        self.assertIsInstance(problem_for_config(config), VRPProblem)

    def test_vrp_evaluator_accepts_complete_metaheuristic_solver(self) -> None:
        instances = load_vrp_instances(
            "data/vrp/test",
            pool_name="test_instances",
            test_sizes=[10],
            test_limit_per_size=2,
        )
        evaluator = VRPCandidateEvaluator(
            instances,
            seeds=[1],
            budget=100,
            timeout_seconds=5,
            timeout_penalty=0.0,
            pool_name="test_instances",
        )
        candidate = Candidate(
            id="cand_vrp",
            generation=0,
            strategy="initial:1",
            name="nearest_balanced",
            code=_SIMPLE_VRP_SOLVER,
            metrics=evaluator.empty_metrics(),
        )

        result = evaluator.evaluate_candidate(candidate)

        self.assertEqual(result.status, "valid")
        self.assertEqual(candidate.status, CandidateStatus.VALID)
        self.assertEqual(result.score_name, "distance")
        self.assertTrue(result.score < float("inf"))
        self.assertEqual(candidate.metrics["problem"], "vrp")

    def test_vrp_validation_requires_metaheuristic_signature(self) -> None:
        valid = validate_vrp_generated_code(_SIMPLE_VRP_SOLVER)
        invalid = validate_vrp_generated_code("def select_next_node(current_node):\n    return 0")

        self.assertTrue(valid.valid)
        self.assertFalse(invalid.valid)

    def test_vrp_metrics_aggregate_route_records(self) -> None:
        metrics = aggregate_vrp_records([
            {
                "status": "valid",
                "gap": 10.0,
                "max_route_distance": 2.0,
                "total_route_distance": 5.0,
                "runtime_seconds": 0.1,
                "dimension": 10,
                "truck_count": 3,
                "source": "fixture",
            },
            {
                "status": "timeout",
                "gap": 20.0,
                "max_route_distance": 3.0,
                "total_route_distance": 6.0,
                "runtime_seconds": 1.0,
                "dimension": 20,
                "truck_count": 3,
                "source": "fixture",
            },
        ], timeout_penalty=4.0)

        self.assertEqual(metrics["runs"], 2)
        self.assertEqual(metrics["valid_count"], 1)
        self.assertEqual(metrics["partial_timeout_count"], 1)
        self.assertEqual(metrics["mean_gap"], 15.0)
        self.assertEqual(metrics["penalized_mean_gap"], 17.0)


_SIMPLE_VRP_SOLVER = r'''
def solve_vrp(distance_matrix, truck_count, seed, budget):
    n = len(distance_matrix)
    routes = [[0, 0] for _ in range(int(truck_count))]
    current = [0 for _ in range(int(truck_count))]
    route_lengths = [0.0 for _ in range(int(truck_count))]
    unvisited = list(range(1, n))
    while unvisited:
        best = None
        for truck in range(int(truck_count)):
            cur = current[truck]
            for node in unvisited:
                increase = float(distance_matrix[cur][node]) + float(distance_matrix[node][0]) - float(distance_matrix[cur][0])
                projected = route_lengths[truck] + increase
                score = max(projected, max(route_lengths[:truck] + route_lengths[truck + 1:] or [0.0]))
                tie = (score, projected, float(distance_matrix[cur][node]), truck, node)
                if best is None or tie < best[0]:
                    best = (tie, truck, node, increase)
        _, truck, node, increase = best
        routes[truck].insert(-1, node)
        current[truck] = node
        route_lengths[truck] += increase
        unvisited.remove(node)
    report_best_vrp(routes)
    return routes
'''


if __name__ == "__main__":
    unittest.main()
