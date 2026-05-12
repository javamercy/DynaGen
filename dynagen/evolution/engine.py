import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from dynagen.candidates import CandidateStatus, ParsedCandidateResponse
from dynagen.candidates.candidate import Candidate
from dynagen.config import RunConfig
from dynagen.evaluation.base import CandidateEvaluator
from dynagen.evolution.population import Population
from dynagen.evolution.selection import select_parents, select_survivors
from dynagen.evolution.strategies import parent_count, Strategy
from dynagen.llm.base import LLMProvider
from dynagen.persistence.run_store import RunStore
from dynagen.problems import problem_for_config
from dynagen.problems.base import Problem
from dynagen.reporting.summary import build_final_report, generation_summary

LLM_REFLECTION_PERIOD = 2


@dataclass(frozen=True)
class _CandidateTask:
    """Immutable descriptor for a single candidate generation task."""
    candidate_id: str
    generation: int
    strategy: str
    parents: list[Candidate]
    messages: list[dict[str, str]]
    prompt: str


class EvolutionEngine:
    def __init__(
            self,
            *,
            config: RunConfig,
            provider: LLMProvider,
            search_evaluator: CandidateEvaluator,
            test_evaluator: CandidateEvaluator,
            store: RunStore,
    ) -> None:
        self.config = config
        self.provider = provider
        self.search_evaluator = search_evaluator
        self.test_evaluator = test_evaluator
        self.store = store
        self.problem: Problem = problem_for_config(config)
        self.rng = random.Random(config.seed)
        self._candidate_index: dict[str, Candidate] = {}

    def run(self) -> Population:
        population = self._initial_population()
        self.store.save_generation(
            0,
            population=population.candidates,
            offspring=[],
            summary=generation_summary(0, population.candidates, []),
        )
        for generation in range(1, self.config.evolution.generations + 1):
            offspring = self._generate_offspring(generation, population)
            next_candidates = select_survivors(population.candidates + offspring, self.config.evolution.population_size)
            population = Population(generation=generation, candidates=next_candidates)
            summary = generation_summary(generation, population.candidates, offspring)
            self.store.save_generation(
                generation,
                population=population.candidates,
                offspring=offspring,
                summary=summary,
            )
        search_best = population.best
        test_result = self.test_evaluator.evaluate_code(search_best.code)
        self.store.save_test_result(search_best.id, test_result)
        llm_calls = self._llm_call_summary()
        self.store.save_llm_calls(llm_calls)
        self.store.write_final_report(
            build_final_report(
                population.candidates,
                search_best=search_best,
                test_result=test_result,
                llm_calls=llm_calls,
            )
        )
        return population

    def _llm_call_summary(self) -> dict:
        summary_getter = getattr(self.provider, "summary", None)
        summary = dict(summary_getter()) if callable(summary_getter) else {}
        configured_budget = scheduled_llm_calls(self.config)
        summary.setdefault("candidate_generation_calls", None)
        summary.setdefault("reflection_calls", None)
        summary.setdefault("total_api_calls", None)
        summary.setdefault("failed_calls", None)
        summary["llm_model"] = getattr(self.provider, "model", None)
        summary["configured_candidate_generation_budget"] = configured_budget
        calls = summary.get("candidate_generation_calls")
        summary["budget_match"] = calls == configured_budget if calls is not None else None
        return summary

    # ------------------------------------------------------------------
    # Initial population: parallel LLM calls + parallel evaluations
    # ------------------------------------------------------------------

    def _initial_population(self) -> Population:
        roles = self.problem.initial_roles(self.config.evolution.population_size)
        tasks = self._build_initial_tasks(roles)
        candidates = self._execute_tasks_parallel(tasks)
        self._register_candidates(candidates)
        return Population.from_candidates(0, candidates, size=self.config.evolution.population_size)

    def _build_initial_tasks(self, roles: list) -> list[_CandidateTask]:
        tasks: list[_CandidateTask] = []
        for role in roles:
            messages = self.problem.build_initial_prompt(role)
            prompt = _format_messages(messages)
            candidate_id = self.store.next_candidate_id()
            tasks.append(_CandidateTask(
                candidate_id=candidate_id,
                generation=0,
                strategy=f"initial:{role.slot}",
                parents=[],
                messages=messages,
                prompt=prompt,
            ))
        return tasks

    # ------------------------------------------------------------------
    # Offspring generation: parallel LLM calls + parallel evaluations
    # ------------------------------------------------------------------

    def _generate_offspring(self, generation: int, population: Population) -> list[Candidate]:
        generation_reflection = ""
        if self._include_llm_reflection(generation):
            generation_reflection = self._generate_llm_reflection(generation, population.candidates)
        tasks = self._build_offspring_tasks(generation, population, generation_reflection=generation_reflection)
        offspring = self._execute_tasks_parallel(tasks)
        self._register_candidates(offspring)
        return offspring

    def _build_offspring_tasks(
            self,
            generation: int,
            population: Population,
            *,
            generation_reflection: str = "",
    ) -> list[_CandidateTask]:
        tasks: list[_CandidateTask] = []
        for strategy in self.config.evolution.strategies:
            for _ in range(self.config.evolution.offspring_per_strategy):
                candidate_id = self.store.next_candidate_id()
                parents = self._select_strategy_parents(strategy, population.candidates)
                messages = self.problem.build_evolution_prompt(
                    strategy,
                    parents,
                    generation_reflection=generation_reflection,
                )
                prompt = _format_messages(messages)
                tasks.append(_CandidateTask(
                    candidate_id=candidate_id,
                    generation=generation,
                    strategy=strategy,
                    parents=parents,
                    messages=messages,
                    prompt=prompt,
                ))
        return tasks

    # ------------------------------------------------------------------
    # Parallel task execution: LLM call → evaluation → persist
    # ------------------------------------------------------------------

    def _execute_tasks_parallel(self, tasks: list[_CandidateTask]) -> list[Candidate]:
        """Execute LLM calls and evaluations concurrently, preserving task order."""
        if not tasks:
            return []

        max_workers = min(len(tasks), 8)
        results: list[Candidate] = [None] * len(tasks)  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_single_task, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()

        return results

    def _process_single_task(self, task: _CandidateTask) -> Candidate:
        """Run LLM call + evaluation for a single candidate. Always returns a Candidate."""
        candidate: Candidate | None = None
        try:
            response = self.provider.complete(
                task.messages,
                temperature=self.config.llm.temperature,
            )
            candidate = _build_candidate_from_response(
                response,
                candidate_id=task.candidate_id,
                generation=task.generation,
                strategy=task.strategy,
                parents=[parent.id for parent in task.parents],
                prompt=task.prompt,
                metrics=self._empty_metrics(),
            )
            self.search_evaluator.evaluate_candidate(candidate)
        except Exception as exc:
            error_details = _exception_details(exc)
            if candidate is None:
                candidate = _failed_candidate(
                    candidate_id=task.candidate_id,
                    generation=task.generation,
                    strategy=task.strategy,
                    parents=[parent.id for parent in task.parents],
                    prompt=task.prompt,
                    error_details=error_details,
                    metrics=self._empty_metrics(),
                )
            else:
                _mark_candidate_error(candidate, error_details)
        self.store.save_candidate(candidate)
        return candidate

    def _select_strategy_parents(self, strategy: Strategy, candidates: list[Candidate]) -> list[Candidate]:
        return select_parents(candidates, parent_count(strategy), self.rng)

    def _empty_metrics(self) -> dict:
        metrics_getter = getattr(self.search_evaluator, "empty_metrics", None)
        return dict(metrics_getter()) if callable(metrics_getter) else {}

    def _include_llm_reflection(self, generation: int) -> bool:
        return self.config.problem.type in {"tsp", "bbob", "dvrp"} and generation > 0 and generation % LLM_REFLECTION_PERIOD == 0

    def _generate_llm_reflection(self, generation: int, candidates: list[Candidate]) -> str:
        prompt_builder = getattr(self.problem, "build_llm_reflection_prompt", None)
        text_completion = getattr(self.provider, "complete_text", None)
        if not callable(prompt_builder) or not callable(text_completion):
            return ""
        candidate = self._select_reflection_candidate(candidates)
        candidate.metrics = candidate.metrics or {}
        reflection = candidate.metrics.setdefault("reflection", {})
        if not isinstance(reflection, dict):
            reflection = {}
            candidate.metrics["reflection"] = reflection
        resolved_parents = self._resolve_parents(candidate)
        prompt = prompt_builder(candidate, parents=resolved_parents, generation=generation)
        reflection_record = {
            "generation": generation,
            "candidate_id": candidate.id,
            "cadence": LLM_REFLECTION_PERIOD,
            "problem": self.config.problem.type,
            "prompt": prompt,
            "model": getattr(self.provider, "model", None),
            "status": "ok",
        }
        try:
            text = text_completion(prompt, temperature=min(self.config.llm.temperature, 0.7))
            normalized_text = " ".join(text.split())[:1200]
            reflection["llm_reflection"] = {
                "generation": generation,
                "cadence": LLM_REFLECTION_PERIOD,
                "candidate_id": candidate.id,
                "text": normalized_text,
            }
            reflection_record["response"] = normalized_text
            self.store.save_candidate(candidate)
            self.store.save_reflection(reflection_record)
            return normalized_text
        except Exception as exc:
            reflection["llm_reflection_error"] = _exception_details(exc)
            reflection_record["status"] = "error"
            reflection_record["error_details"] = _exception_details(exc)
            self.store.save_reflection(reflection_record)
            self.store.save_candidate(candidate)
            return ""

    def _select_reflection_candidate(self, candidates: list[Candidate]) -> Candidate:
        ordered = select_survivors(candidates, len(candidates))
        for candidate in ordered:
            if self._resolve_parents(candidate):
                return candidate
        return ordered[0]

    def _resolve_parents(self, candidate: Candidate) -> list[Candidate]:
        return [
            self._candidate_index[parent_id]
            for parent_id in candidate.parents
            if parent_id in self._candidate_index
        ]

    def _register_candidates(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            self._candidate_index[candidate.id] = candidate


def scheduled_llm_calls(config: RunConfig) -> int:
    return (
            config.evolution.population_size
            + config.evolution.generations
            * len(config.evolution.strategies)
            * config.evolution.offspring_per_strategy
    )


def _build_candidate_from_response(
        response: ParsedCandidateResponse,
        *,
        candidate_id: str,
        generation: int,
        strategy: str,
        parents: list[str] | None = None,
        prompt: str,
        metrics: dict | None = None,
    ) -> Candidate:
    candidate_metrics = dict(metrics) if metrics is not None else {}
    uses_distance = _uses_distance_metrics(candidate_metrics)
    if uses_distance:
        candidate_metrics.setdefault("score_name", "distance")
        candidate_metrics["distance"] = math.inf
    return Candidate(
        id=candidate_id,
        generation=generation,
        strategy=strategy,
        name=response.name,
        thought=response.thought,
        code=response.code,
        parents=list(parents or []),
        fitness=None if uses_distance else None,
        distance=math.inf if uses_distance else None,
        metrics=candidate_metrics,
        status=CandidateStatus.PENDING,
        prompt=prompt,
    )


def _failed_candidate(
        *,
        candidate_id: str,
        generation: int,
        strategy: str,
        prompt: str,
        parents: list[str] | None = None,
        error_details: str | None = None,
        metrics: dict | None = None,
) -> Candidate:
    candidate_metrics = dict(metrics) if metrics is not None else {}
    uses_distance = _uses_distance_metrics(candidate_metrics)
    if uses_distance:
        candidate_metrics.setdefault("score_name", "distance")
        candidate_metrics["distance"] = math.inf
    return Candidate(
        id=candidate_id,
        generation=generation,
        strategy=strategy,
        name="",
        thought="",
        code="",
        parents=list(parents or []),
        fitness=None if uses_distance else math.inf,
        distance=math.inf if uses_distance else None,
        metrics=candidate_metrics,
        status=CandidateStatus.ERROR,
        prompt=prompt,
        error_details=error_details,
    )


def _exception_details(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _mark_candidate_error(candidate: Candidate, error_details: str) -> None:
    candidate.status = CandidateStatus.ERROR
    if not candidate.metrics:
        candidate.metrics = {}
    if candidate.score_name == "distance":
        candidate.distance = math.inf
        candidate.fitness = None
        candidate.metrics.setdefault("score_name", "distance")
        candidate.metrics["distance"] = math.inf
    else:
        candidate.fitness = math.inf
    candidate.error_details = error_details


def _uses_distance_metrics(metrics: dict) -> bool:
    return (
        metrics.get("problem") == "tsp"
        or metrics.get("problem") == "dvrp"
        or metrics.get("score_name") == "distance"
        or "distance" in metrics
    )


def _format_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{message['role']}]\n{message['content']}"
        for message in messages
    )
