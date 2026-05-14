import json
import tempfile
import unittest

from dynagen.candidates import CandidateStatus, ParsedCandidateResponse
from dynagen.candidates.candidate import Candidate
from dynagen.config import RunConfig
from dynagen.evaluation.bbob_archive import build_bbob_archive_profile
from dynagen.evaluation.base import EvaluationResult
from dynagen.evaluation.dvrp_archive import build_dvrp_archive_profile
from dynagen.evaluation.tsp_archive import build_tsp_archive_profile
from dynagen.evolution.archive import CandidateArchive
from dynagen.persistence.run_store import RunStore
from dynagen.evolution.engine import EvolutionEngine


class ArchiveTests(unittest.TestCase):
    def test_archive_config_parses_nested_options(self) -> None:
        config = _run_config(population_size=1, generations=0)

        self.assertTrue(config.evolution.archive.enabled)
        self.assertEqual(config.evolution.archive.max_size, 8)
        self.assertEqual(config.evolution.archive.max_per_bucket, 2)
        self.assertEqual(config.evolution.archive.parent_sample_probability, 1.0)
        self.assertEqual(config.evolution.archive.s3_archive_parent_min, 1)

    def test_tsp_archive_profile_uses_size_and_mechanism_buckets(self) -> None:
        candidate = Candidate(
            id="cand_1",
            generation=0,
            strategy="initial:1",
            name="solver",
            thought="",
            code="def solve_tsp(distance_matrix, seed, budget):\n    # 2-opt nearest insertion restart\n    pass",
            distance=100.0,
            status=CandidateStatus.VALID,
            metrics={
                "problem": "tsp",
                "score_name": "distance",
                "distance": 100.0,
                "runs": 2,
                "valid_count": 2,
                "mean_tour_length": 100.0,
                "timeout_fraction": 0.0,
                "mean_runtime": 0.1,
                "score_by_instance_size": {"33": 80.0, "201": 140.0},
                "score_by_instance_source": {"synthetic:llamea:11:32": 80.0},
            },
        )

        profile = build_tsp_archive_profile(candidate)

        self.assertIn("global", profile["buckets"])
        self.assertIn("tsp:size:33", profile["buckets"])
        self.assertIn("tsp:size:201", profile["buckets"])
        self.assertIn("tsp:mechanism:two_opt", profile["buckets"])
        self.assertIn("tsp:runtime:robust", profile["buckets"])
        self.assertEqual(profile["primary_bucket"], "tsp:size:33")

    def test_bbob_archive_profile_uses_group_and_mechanism_buckets(self) -> None:
        candidate = Candidate(
            id="cand_1",
            generation=0,
            strategy="initial:1",
            name="optimizer",
            thought="",
            code="class Optimizer:\n    # population restart coordinate search\n    pass",
            fitness=0.2,
            status=CandidateStatus.VALID,
            metrics={
                "problem": "bbob",
                "runs": 2,
                "valid_count": 2,
                "mean_aocc": 0.8,
                "penalized_mean_aocc": 0.8,
                "mean_final_error": 0.1,
                "timeout_fraction": 0.0,
                "mean_runtime": 0.1,
                "aocc_by_group": {"separable": 0.9, "multimodal": 0.5},
                "aocc_by_function": {"1": 0.9},
            },
        )

        profile = build_bbob_archive_profile(candidate)

        self.assertIn("bbob:group:separable", profile["buckets"])
        self.assertIn("bbob:function:1", profile["buckets"])
        self.assertIn("bbob:mechanism:evolution_strategy", profile["buckets"])
        self.assertIn("bbob:mechanism:restart", profile["buckets"])

    def test_dvrp_archive_profile_uses_size_truck_and_behavior_buckets(self) -> None:
        candidate = Candidate(
            id="cand_1",
            generation=0,
            strategy="initial:1",
            name="dispatch",
            thought="",
            code="def choose_next_customer(*args):\n    # nearest wait depot truck balance\n    return None",
            distance=10.0,
            status=CandidateStatus.VALID,
            metrics={
                "problem": "dvrp",
                "score_name": "distance",
                "distance": 10.0,
                "runs": 2,
                "valid_count": 2,
                "mean_gap": 10.0,
                "worst_gap": 20.0,
                "mean_makespan": 100.0,
                "mean_waits": 1.0,
                "mean_completed_count": 10.0,
                "timeout_fraction": 0.0,
                "mean_runtime": 0.1,
                "score_by_instance_size": {"10": 8.0, "20": 12.0},
                "score_by_truck_count": {"2": 8.0},
                "score_by_instance_source": {"paper_train": 8.0},
            },
        )

        profile = build_dvrp_archive_profile(candidate)

        self.assertIn("dvrp:size:10", profile["buckets"])
        self.assertIn("dvrp:trucks:2", profile["buckets"])
        self.assertIn("dvrp:waits:low", profile["buckets"])
        self.assertIn("dvrp:mechanism:nearest_available", profile["buckets"])

    def test_archive_rejects_duplicate_code_when_weaker(self) -> None:
        archive = CandidateArchive(config=_run_config(population_size=1, generations=0).evolution.archive, problem="tsp")
        strong = _candidate("cand_1", score=10.0, code="def solve_tsp(a,b,c):\n    return []")
        weak = _candidate("cand_2", score=20.0, code="def solve_tsp(a,b,c):\n    return []")

        archive.update([strong], generation=0, profile_builder=build_tsp_archive_profile)
        archive.update([weak], generation=0, profile_builder=build_tsp_archive_profile)

        self.assertIn("cand_1", archive.entries)
        self.assertNotIn("cand_2", archive.entries)
        self.assertEqual(archive.stats["rejected_duplicate_count"], 1)

    def test_engine_samples_archive_parent_and_persists_summary(self) -> None:
        provider = _FakeProvider()
        evaluator = _FakeEvaluator()
        config = _run_config(population_size=1, generations=1, strategies=["S1"])

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(tmpdir)
            EvolutionEngine(
                config=config,
                provider=provider,
                search_evaluator=evaluator,
                test_evaluator=evaluator,
                store=store,
            ).run()

            prompt = (store.prompts_dir / "cand_000002_prompt.txt").read_text(encoding="utf-8")
            archive_summary = json.loads((store.root / "archive_summary.json").read_text(encoding="utf-8"))
            generation_summary = json.loads(
                (store.generations_dir / "generation_001" / "summary.json").read_text(encoding="utf-8")
            )
            llm_calls = json.loads((store.root / "llm_calls.json").read_text(encoding="utf-8"))

        self.assertIn("Archive source: yes", prompt)
        self.assertGreaterEqual(archive_summary["size"], 1)
        self.assertIn("archive", generation_summary)
        self.assertIn("archive", llm_calls)
        self.assertGreaterEqual(llm_calls["archive"]["parent_selections_from_archive"], 1)

    def test_archive_disabled_omits_archive_parent_context(self) -> None:
        provider = _FakeProvider()
        evaluator = _FakeEvaluator()
        config = _run_config(population_size=1, generations=1, strategies=["S1"], archive_enabled=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(tmpdir)
            EvolutionEngine(
                config=config,
                provider=provider,
                search_evaluator=evaluator,
                test_evaluator=evaluator,
                store=store,
            ).run()

            prompt = (store.prompts_dir / "cand_000002_prompt.txt").read_text(encoding="utf-8")
            llm_calls = json.loads((store.root / "llm_calls.json").read_text(encoding="utf-8"))

        self.assertNotIn("Archive source: yes", prompt)
        self.assertFalse(llm_calls["archive"]["enabled"])


def _candidate(candidate_id: str, *, score: float, code: str) -> Candidate:
    return Candidate(
        id=candidate_id,
        generation=0,
        strategy="initial:1",
        name="solver",
        thought="",
        code=code,
        distance=score,
        status=CandidateStatus.VALID,
        metrics={
            "problem": "tsp",
            "score_name": "distance",
            "distance": score,
            "runs": 1,
            "valid_count": 1,
            "mean_tour_length": score,
            "timeout_fraction": 0.0,
            "mean_runtime": 0.1,
            "score_by_instance_size": {"33": score},
            "score_by_instance_source": {"synthetic:llamea:11:32": score},
        },
    )


class _FakeProvider:
    def __init__(self) -> None:
        self.model = "fake"
        self.calls = 0

    def complete(self, messages, *, temperature):
        self.calls += 1
        return ParsedCandidateResponse(
            name=f"solver_{self.calls}",
            thought="fake solver",
            code=(
                "def solve_tsp(distance_matrix, seed, budget):\n"
                f"    tag = {self.calls}\n"
                "    return list(range(len(distance_matrix)))"
            ),
        )

    def complete_with_metadata(self, messages, *, temperature):
        raise NotImplementedError

    def summary(self):
        return {
            "candidate_generation_calls": self.calls,
            "feedback_calls": 0,
            "reflection_calls": 0,
            "total_api_calls": self.calls,
            "failed_calls": 0,
            "configured_candidate_generation_budget": None,
            "budget_match": None,
            "llm_model": self.model,
        }


class _FakeEvaluator:
    def empty_metrics(self):
        return {"problem": "tsp", "score_name": "distance", "distance": None}

    def evaluate_candidate(self, candidate):
        result = self.evaluate_code(candidate.code)
        candidate.status = CandidateStatus(result.status)
        candidate.distance = result.score
        candidate.metrics = dict(result.metrics)
        candidate.metrics["distance"] = result.score
        candidate.error_details = result.error_feedback
        return result

    def evaluate_code(self, code):
        score = 10.0 if "tag = 1" in code else 11.0
        metrics = {
            "problem": "tsp",
            "score_name": "distance",
            "distance": score,
            "runs": 1,
            "valid_count": 1,
            "mean_tour_length": score,
            "mean_gap": None,
            "median_gap": None,
            "worst_gap": None,
            "best_gap": None,
            "timeout_fraction": 0.0,
            "mean_runtime": 0.1,
            "score_by_instance_size": {"33": score},
            "score_by_instance_source": {"synthetic:llamea:11:32": score},
        }
        return EvaluationResult("valid", score, metrics, score_name="distance")


def _run_config(
        *,
        population_size: int,
        generations: int,
        strategies: list[str] | None = None,
        archive_enabled: bool = True,
) -> RunConfig:
    return RunConfig.from_dict({
        "run": {"name": "archive_test", "output_dir": "runs/test", "seed": 1},
        "llm": {
            "provider": "ollama",
            "model": "fake",
            "temperature": 0.1,
        },
        "evolution": {
            "population_size": population_size,
            "generations": generations,
            "offspring_per_strategy": 1,
            "strategies": strategies or ["S1"],
            "archive": {
                "enabled": archive_enabled,
                "max_size": 8,
                "max_per_bucket": 2,
                "parent_sample_probability": 1.0,
                "s3_archive_parent_min": 1,
                "final_selection_uses_archive": True,
                "deduplicate_code": True,
            },
        },
        "evaluation": {
            "budget": 10,
            "timeout_seconds": 1,
            "seeds": [1],
            "metric": "mean_gap",
        },
        "data": {
            "search_instances": "unused",
            "test_instances": "unused",
        },
    })


if __name__ == "__main__":
    unittest.main()
