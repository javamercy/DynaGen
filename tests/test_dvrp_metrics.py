import unittest

from dynagen.candidates import CandidateStatus
from dynagen.candidates.candidate import Candidate
from dynagen.evaluation.dvrp_metrics import aggregate_dvrp_records


class DVRPMetricTests(unittest.TestCase):
    def test_aggregate_uses_ttt_as_primary_score(self) -> None:
        records = [
            {
                "status": "valid",
                "ttt": 12.0,
                "gap": 20.0,
                "decisions": 3,
                "waits": 1,
                "completed_count": 10,
                "runtime_seconds": 0.1,
                "dimension": 11,
                "truck_count": 2,
                "source": "synthetic",
            }
        ]

        metrics = aggregate_dvrp_records(records, timeout_penalty=0.0)

        self.assertEqual(metrics["mean_ttt"], 12.0)
        self.assertEqual(metrics["mean_gap"], 20.0)
        self.assertEqual(metrics["score_by_instance_size"], {"11": 12.0})
        self.assertEqual(metrics["gap_by_instance_size"], {"11": 20.0})
        self.assertNotIn("mean_makespan", metrics)
        self.assertNotIn("timeout_distance", metrics)

    def test_dvrp_candidate_serializes_score_as_ttt(self) -> None:
        candidate = Candidate(
            id="greedy",
            generation=0,
            strategy="baseline",
            distance=49.8579,
            status=CandidateStatus.VALID,
            metrics={"problem": "dvrp", "score_name": "ttt", "ttt": 49.8579},
        )

        data = candidate.to_dict(include_code=False)

        self.assertEqual(candidate.score_name, "ttt")
        self.assertEqual(data["ttt"], 49.8579)
        self.assertNotIn("distance", data)


if __name__ == "__main__":
    unittest.main()
