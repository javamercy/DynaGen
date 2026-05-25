import json
import tempfile
import unittest

from dynagen.candidates import CandidateStatus, ParsedCandidateResponse
from dynagen.candidates.candidate import Candidate
from dynagen.config import RunConfig
from dynagen.evaluation.bbob_history import build_bbob_history_profile
from dynagen.evaluation.base import EvaluationResult
from dynagen.evaluation.dvrp_history import build_dvrp_history_profile
from dynagen.evaluation.tsp_history import build_tsp_history_profile
from dynagen.evaluation.vrp_history import build_vrp_history_profile
from dynagen.evolution.history import CandidateHistory
from dynagen.persistence.run_store import RunStore
from dynagen.evolution.engine import EvolutionEngine


class HistoryTests(unittest.TestCase):
    def test_history_config_parses_nested_options(self) -> None:
        config = _run_config(population_size=1, generations=0)

        self.assertTrue(config.evolution.history.enabled)
        self.assertEqual(config.evolution.history.max_size, 8)
        self.assertEqual(config.evolution.history.max_per_bucket, 2)
        self.assertEqual(config.evolution.history.parent_sample_probability, 1.0)
        self.assertEqual(config.evolution.history.s3_history_parent_min, 1)

    def test_tsp_history_profile_uses_size_and_mechanism_buckets(self) -> None:
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
                "score_by_instance_source": {"synthetic:tsp_construct:n_instance=11:n_cities=32:seed=11": 80.0},
            },
        )

        profile = build_tsp_history_profile(candidate)

        self.assertIn("global", profile["buckets"])
        self.assertIn("tsp:size:33", profile["buckets"])
        self.assertIn("tsp:size:201", profile["buckets"])
        self.assertIn("tsp:mechanism:two_opt", profile["buckets"])
        self.assertIn("tsp:runtime:robust", profile["buckets"])
        self.assertEqual(profile["primary_bucket"], "tsp:size:33")

    def test_bbob_history_profile_uses_group_and_mechanism_buckets(self) -> None:
        candidate = Candidate(
            id="cand_1",
            generation=0,
            strategy="initial:1",
            name="optimizer",
            thought="",
            code="class Optimizer:\n    # population restart coordinate search\n    pass",
            status=CandidateStatus.VALID,
            metrics={
                "problem": "bbob",
                "score_name": "mean_aocc",
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

        profile = build_bbob_history_profile(candidate)

        self.assertIn("bbob:group:separable", profile["buckets"])
        self.assertIn("bbob:function:1", profile["buckets"])
        self.assertIn("bbob:mechanism:evolution_strategy", profile["buckets"])
        self.assertIn("bbob:mechanism:restart", profile["buckets"])

    def test_dvrp_history_profile_uses_size_truck_and_behavior_buckets(self) -> None:
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
                "score_name": "ttt",
                "ttt": 10.0,
                "runs": 2,
                "valid_count": 2,
                "mean_gap": 10.0,
                "worst_gap": 20.0,
                "mean_ttt": 100.0,
                "mean_waits": 1.0,
                "mean_completed_count": 10.0,
                "timeout_fraction": 0.0,
                "mean_runtime": 0.1,
                "score_by_instance_size": {"10": 8.0, "20": 12.0},
                "score_by_truck_count": {"2": 8.0},
                "score_by_instance_source": {"paper_train": 8.0},
            },
        )

        profile = build_dvrp_history_profile(candidate)

        self.assertIn("dvrp:size:10", profile["buckets"])
        self.assertIn("dvrp:trucks:2", profile["buckets"])
        self.assertIn("dvrp:waits:low", profile["buckets"])
        self.assertIn("dvrp:mechanism:nearest_available", profile["buckets"])

    def test_vrp_history_profile_uses_size_truck_and_mechanism_buckets(self) -> None:
        candidate = Candidate(
            id="cand_1",
            generation=0,
            strategy="initial:1",
            name="vrp_solver",
            thought="",
            code="def solve_vrp(*args):\n    # sweep savings two_opt balance restart\n    return []",
            distance=10.0,
            status=CandidateStatus.VALID,
            metrics={
                "problem": "vrp",
                "score_name": "gap",
                "gap": 10.0,
                "runs": 2,
                "valid_count": 2,
                "mean_gap": 10.0,
                "worst_gap": 20.0,
                "mean_max_route_distance": 2.5,
                "mean_total_route_distance": 8.0,
                "timeout_fraction": 0.0,
                "mean_runtime": 0.1,
                "score_by_instance_size": {"10": 8.0, "20": 12.0},
                "score_by_truck_count": {"3": 8.0},
                "score_by_instance_source": {"vrp_train": 8.0},
            },
        )

        profile = build_vrp_history_profile(candidate)

        self.assertIn("vrp:size:10", profile["buckets"])
        self.assertIn("vrp:trucks:3", profile["buckets"])
        self.assertIn("vrp:mechanism:sweep", profile["buckets"])
        self.assertIn("vrp:mechanism:savings", profile["buckets"])

    def test_history_rejects_duplicate_code_when_weaker(self) -> None:
        history = CandidateHistory(config=_run_config(population_size=1, generations=0).evolution.history, problem="tsp")
        strong = _candidate("cand_1", score=10.0, code="def solve_tsp(a,b,c):\n    return []")
        weak = _candidate("cand_2", score=20.0, code="def solve_tsp(a,b,c):\n    return []")

        history.update([strong], generation=0, profile_builder=build_tsp_history_profile)
        history.update([weak], generation=0, profile_builder=build_tsp_history_profile)

        self.assertIn("cand_1", history.entries)
        self.assertNotIn("cand_2", history.entries)
        self.assertEqual(history.stats["rejected_duplicate_count"], 1)

    def test_engine_samples_history_parent_and_persists_summary(self) -> None:
        provider = _FakeProvider()
        evaluator = _FakeEvaluator()
        config = _run_config(population_size=1, generations=1, strategies=["e1_radical_exploration"])

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
            history_summary = json.loads((store.root / "history_summary.json").read_text(encoding="utf-8"))
            generation_summary = json.loads(
                (store.generations_dir / "generation_001" / "summary.json").read_text(encoding="utf-8")
            )
            llm_calls = json.loads((store.root / "llm_calls.json").read_text(encoding="utf-8"))

        self.assertIn("History source: yes", prompt)
        self.assertGreaterEqual(history_summary["size"], 1)
        self.assertIn("history", generation_summary)
        self.assertIn("history", llm_calls)
        self.assertGreaterEqual(llm_calls["history"]["parent_selections_from_history"], 1)

    def test_history_disabled_omits_history_parent_context(self) -> None:
        provider = _FakeProvider()
        evaluator = _FakeEvaluator()
        config = _run_config(population_size=1, generations=1, strategies=["e1_radical_exploration"], history_enabled=False)

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

        self.assertNotIn("History source: yes", prompt)
        self.assertFalse(llm_calls["history"]["enabled"])


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
            "score_by_instance_source": {"synthetic:tsp_construct:n_instance=11:n_cities=32:seed=11": score},
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
            "score_by_instance_source": {"synthetic:tsp_construct:n_instance=11:n_cities=32:seed=11": score},
        }
        return EvaluationResult("valid", score, metrics, score_name="distance")


def _run_config(
        *,
        population_size: int,
        generations: int,
        strategies: list[str] | None = None,
        history_enabled: bool = True,
) -> RunConfig:
    return RunConfig.from_dict({
        "run": {"name": "history_test", "output_dir": "runs/test", "seed": 1},
        "llm": {
            "provider": "ollama",
            "model": "fake",
            "temperature": 0.1,
        },
        "evolution": {
            "population_size": population_size,
            "generations": generations,
            "offspring_per_strategy": 1,
            "strategies": strategies or ["e1_radical_exploration"],
            "history": {
                "enabled": history_enabled,
                "max_size": 8,
                "max_per_bucket": 2,
                "parent_sample_probability": 1.0,
                "s3_history_parent_min": 1,
                "final_selection_uses_history": True,
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
