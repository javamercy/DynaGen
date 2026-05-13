import json
import tempfile
import unittest

from dynagen.candidates import CandidateStatus, ParsedCandidateResponse
from dynagen.candidates.candidate import Candidate
from dynagen.config import RunConfig
from dynagen.evaluation.base import EvaluationResult
from dynagen.evaluation.tsp_gradient import build_tsp_static_verbal_gradient
from dynagen.evolution.engine import EvolutionEngine
from dynagen.evolution.verbal_gradient import (
    VERBAL_GRADIENT_KEY,
    format_parent_verbal_gradients,
    get_candidate_gradient,
)
from dynagen.persistence.run_store import RunStore


class VerbalGradientTests(unittest.TestCase):
    def test_config_parses_nested_verbal_gradient_options(self) -> None:
        config = _run_config(llm_enabled=True)

        self.assertTrue(config.evolution.verbal_gradients.enabled)
        self.assertTrue(config.evolution.verbal_gradients.llm_enabled)
        self.assertEqual(config.evolution.verbal_gradients.max_llm_calls_per_generation, 1)
        self.assertEqual(config.evolution.verbal_gradients.max_chars, 900)

    def test_tsp_static_gradient_records_timeout_weakness(self) -> None:
        candidate = Candidate(
            id="cand_1",
            generation=1,
            strategy="S1",
            name="timeout_solver",
            thought="",
            code="",
            distance=20.0,
            status=CandidateStatus.TIMEOUT,
            metrics={
                "problem": "tsp",
                "score_name": "distance",
                "distance": 20.0,
                "mean_gap": 12.0,
                "worst_gap": 30.0,
                "timeout_fraction": 0.5,
                "score_by_instance_size": {"50": 8.0, "200": 18.0},
            },
        )

        gradient = build_tsp_static_verbal_gradient(candidate, parents=[], generation=1)

        self.assertEqual(gradient["problem"], "tsp")
        self.assertEqual(gradient["source"], "static")
        self.assertIn("S2", gradient["next_mutations"])
        self.assertTrue(any("timed out" in weakness for weakness in gradient["weaknesses"]))

    def test_parent_gradient_formatting_is_strategy_specific(self) -> None:
        candidate = Candidate(
            id="cand_1",
            generation=0,
            strategy="initial:1",
            name="solver",
            metrics={
                "problem": "tsp",
                "score_name": "distance",
                VERBAL_GRADIENT_KEY: {
                    "source": "static",
                    "summary": "Good incumbent, weak large instances.",
                    "preserve": ["early reporting"],
                    "weaknesses": ["large instances"],
                    "next_mutations": {"S2": "Add a guarded late-budget local pass."},
                    "avoid": ["unbounded loops"],
                },
            },
            distance=10.0,
            status=CandidateStatus.VALID,
        )

        text = format_parent_verbal_gradients([candidate], strategy="S2", max_chars=1000)

        self.assertIn("PARENT-SPECIFIC VERBAL GRADIENTS", text)
        self.assertIn("Next S2 mutation", text)
        self.assertIn("guarded late-budget", text)

    def test_engine_attaches_static_and_cached_llm_gradients(self) -> None:
        provider = _FakeProvider()
        search_evaluator = _FakeEvaluator()
        test_evaluator = _FakeEvaluator()
        config = _run_config(llm_enabled=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(tmpdir)
            EvolutionEngine(
                config=config,
                provider=provider,
                search_evaluator=search_evaluator,
                test_evaluator=test_evaluator,
                store=store,
            ).run()

            initial = store.load_candidate("cand_000001")
            offspring = store.load_candidate("cand_000002")
            initial_prompt = (store.prompts_dir / "cand_000002_prompt.txt").read_text(encoding="utf-8")

        self.assertEqual(provider.text_calls, 1)
        self.assertEqual(get_candidate_gradient(initial)["source"], "static+llm")
        self.assertEqual(get_candidate_gradient(offspring)["source"], "static")
        self.assertIn("PARENT-SPECIFIC VERBAL GRADIENTS", initial_prompt)


class _FakeProvider:
    def __init__(self) -> None:
        self.candidate_calls = 0
        self.text_calls = 0

    def complete(self, messages, *, temperature):
        self.candidate_calls += 1
        return ParsedCandidateResponse(
            name=f"solver_{self.candidate_calls}",
            thought="fake solver",
            code="def solve_tsp(distance_matrix, seed, budget):\n    return list(range(len(distance_matrix)))",
        )

    def complete_with_metadata(self, messages, *, temperature):
        raise NotImplementedError

    def complete_text(self, messages, *, temperature):
        self.text_calls += 1
        return json.dumps({
            "summary": "LLM-targeted parent guidance.",
            "preserve": ["early incumbent reporting"],
            "weaknesses": ["large instances"],
            "next_mutations": {
                "S1": "Explore a different construction.",
                "S2": "Refine the local pass.",
                "S3": "Use only the construction mechanism.",
                "default": "Improve robustness.",
            },
            "avoid": ["unbounded loops"],
        })


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
        metrics = {
            "problem": "tsp",
            "score_name": "distance",
            "distance": 10.0,
            "runs": 1,
            "valid_count": 1,
            "mean_tour_length": 10.0,
            "mean_gap": 5.0,
            "median_gap": 5.0,
            "worst_gap": 5.0,
            "best_gap": 5.0,
            "timeout_fraction": 0.0,
            "score_by_instance_size": {"4": 5.0},
            "score_by_instance_source": {"fake": 5.0},
        }
        return EvaluationResult("valid", 10.0, metrics, score_name="distance")


def _run_config(*, llm_enabled: bool) -> RunConfig:
    return RunConfig.from_dict({
        "run": {"name": "test", "output_dir": "runs/test", "seed": 1},
        "llm": {
            "provider": "ollama",
            "model": "fake",
            "temperature": 0.1,
        },
        "evolution": {
            "population_size": 1,
            "generations": 1,
            "offspring_per_strategy": 1,
            "strategies": ["S1"],
            "verbal_gradients": {
                "enabled": True,
                "static_enabled": True,
                "llm_enabled": llm_enabled,
                "max_llm_calls_per_generation": 1,
                "temperature": 0.2,
                "max_chars": 900,
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
