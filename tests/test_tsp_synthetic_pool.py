import unittest

from dynagen.domain.tsp_synthetic import generate_tsp_construct_instances, parse_tsp_construct_spec
from dynagen.evaluation.tsp_metrics import aggregate_tsp_records
from dynagen.problems.tsp import load_tsp_instances


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


if __name__ == "__main__":
    unittest.main()
