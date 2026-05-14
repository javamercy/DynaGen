import unittest

from dynagen.domain.tsp_synthetic import parse_llamea_tsp_spec, parse_llamea_tsp_specs
from dynagen.evaluation.tsp_metrics import aggregate_tsp_records
from dynagen.problems.tsp import load_tsp_instances


class TSPSyntheticPoolTests(unittest.TestCase):
    def test_single_llamea_spec_remains_supported(self) -> None:
        self.assertEqual(parse_llamea_tsp_spec("synthetic:llamea:69:32"), (69, 32))
        self.assertEqual(parse_llamea_tsp_specs("synthetic:llamea:69:32"), [(69, 32)])

    def test_multi_llamea_spec_expands_seed_size_grid(self) -> None:
        specs = parse_llamea_tsp_specs("synthetic:llamea:seeds=11,23,69:sizes=32,50,100,200")

        self.assertEqual(len(specs), 12)
        self.assertEqual(specs[0], (11, 32))
        self.assertEqual(specs[-1], (69, 200))

    def test_loader_generates_multi_instance_pool_in_dynagen(self) -> None:
        instances = load_tsp_instances("synthetic:llamea:seeds=11,23,69:sizes=32,50,100,200")

        self.assertEqual(len(instances), 12)
        self.assertEqual(instances[0].name, "llamea_seed11_size32")
        self.assertEqual(instances[0].dimension, 33)
        self.assertEqual(instances[-1].name, "llamea_seed69_size200")
        self.assertEqual(instances[-1].dimension, 201)
        self.assertEqual(instances[-1].metadata["source"], "synthetic:llamea:69:200")

    def test_synthetic_group_scores_fall_back_to_tour_length_without_gap(self) -> None:
        metrics = aggregate_tsp_records([
            {
                "status": "valid",
                "tour_length": 10.0,
                "gap": None,
                "dimension": 33,
                "source": "synthetic:llamea:11:32",
                "runtime_seconds": 0.1,
            },
            {
                "status": "valid",
                "tour_length": 14.0,
                "gap": None,
                "dimension": 33,
                "source": "synthetic:llamea:23:32",
                "runtime_seconds": 0.1,
            },
            {
                "status": "valid",
                "tour_length": 40.0,
                "gap": None,
                "dimension": 51,
                "source": "synthetic:llamea:11:50",
                "runtime_seconds": 0.1,
            },
        ])

        self.assertEqual(metrics["score_by_instance_size"], {"33": 12.0, "51": 40.0})
        self.assertEqual(metrics["gap_by_instance_size"], {"33": None, "51": None})
        self.assertEqual(metrics["tour_length_by_instance_size"], {"33": 12.0, "51": 40.0})


if __name__ == "__main__":
    unittest.main()
