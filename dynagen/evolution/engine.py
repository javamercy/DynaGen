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
from dynagen.evolution.verbal_gradient import (
    candidate_has_llm_gradient,
    format_parent_verbal_gradients,
    get_candidate_gradient,
    parse_llm_verbal_gradient,
    set_candidate_gradient,
)
from dynagen.llm.base import LLMProvider
from dynagen.persistence.run_store import RunStore
from dynagen.problems import problem_for_config
from dynagen.problems.base import Problem
from dynagen.reporting.summary import build_final_report, generation_summary


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
        self._llm_gradient_calls_by_generation: dict[int, int] = {}
        self._verbal_gradient_stats: dict[str, int] = {
            "static_count": 0,
            "llm_count": 0,
            "llm_error_count": 0,
        }

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
        feedback_calls = summary.get("feedback_calls", summary.get("reflection_calls"))
        if feedback_calls is None:
            feedback_calls = self._verbal_gradient_stats["llm_count"] + self._verbal_gradient_stats["llm_error_count"]
        summary["feedback_calls"] = feedback_calls
        if summary.get("reflection_calls") is None:
            summary["reflection_calls"] = feedback_calls
        summary["verbal_gradients"] = {
            "enabled": self.config.evolution.verbal_gradients.enabled,
            "static_enabled": self.config.evolution.verbal_gradients.static_enabled,
            "llm_enabled": self.config.evolution.verbal_gradients.llm_enabled,
            **self._verbal_gradient_stats,
        }
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
        tasks = self._build_offspring_tasks(generation, population)
        offspring = self._execute_tasks_parallel(tasks)
        self._register_candidates(offspring)
        return offspring

    def _build_offspring_tasks(
            self,
            generation: int,
            population: Population,
    ) -> list[_CandidateTask]:
        tasks: list[_CandidateTask] = []
        for strategy in self.config.evolution.strategies:
            for _ in range(self.config.evolution.offspring_per_strategy):
                candidate_id = self.store.next_candidate_id()
                parents = self._select_strategy_parents(strategy, population.candidates)
                self._ensure_parent_verbal_gradients(strategy, parents, generation)
                feedback_context = self._feedback_context(strategy, parents)
                messages = self.problem.build_evolution_prompt(
                    strategy,
                    parents,
                    feedback_context=feedback_context,
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
        self._attach_static_verbal_gradient(candidate, task.parents, task.generation)
        self.store.save_candidate(candidate)
        return candidate

    def _select_strategy_parents(self, strategy: Strategy, candidates: list[Candidate]) -> list[Candidate]:
        return select_parents(candidates, parent_count(strategy), self.rng)

    def _empty_metrics(self) -> dict:
        metrics_getter = getattr(self.search_evaluator, "empty_metrics", None)
        return dict(metrics_getter()) if callable(metrics_getter) else {}

    def _attach_static_verbal_gradient(
            self,
            candidate: Candidate,
            parents: list[Candidate],
            generation: int,
    ) -> dict | None:
        gradient_config = self.config.evolution.verbal_gradients
        if not gradient_config.enabled or not gradient_config.static_enabled:
            return None
        if get_candidate_gradient(candidate):
            return get_candidate_gradient(candidate)
        gradient = self._build_static_verbal_gradient(candidate, parents, generation)
        if gradient is None:
            return None
        set_candidate_gradient(candidate, gradient)
        self._verbal_gradient_stats["static_count"] += 1
        return gradient

    def _build_static_verbal_gradient(
            self,
            candidate: Candidate,
            parents: list[Candidate],
            generation: int,
    ) -> dict | None:
        builder = getattr(self.problem, "build_static_verbal_gradient", None)
        if not callable(builder):
            return None
        return dict(builder(candidate, parents=parents, generation=generation))

    def _ensure_parent_verbal_gradients(
            self,
            strategy: Strategy,
            parents: list[Candidate],
            generation: int,
    ) -> None:
        gradient_config = self.config.evolution.verbal_gradients
        if not gradient_config.enabled:
            return
        for parent in parents:
            parent_parents = self._resolve_parents(parent)
            static_gradient = get_candidate_gradient(parent)
            if static_gradient is None and gradient_config.static_enabled:
                static_gradient = self._attach_static_verbal_gradient(parent, parent_parents, parent.generation)
                self.store.save_candidate(parent)
            if not gradient_config.llm_enabled:
                continue
            if candidate_has_llm_gradient(parent):
                continue
            if self._llm_gradient_calls_by_generation.get(generation, 0) >= gradient_config.max_llm_calls_per_generation:
                continue
            if static_gradient is None:
                static_gradient = self._build_static_verbal_gradient(parent, parent_parents, parent.generation)
            if static_gradient is None:
                continue
            self._generate_llm_verbal_gradient(
                candidate=parent,
                parents=parent_parents,
                generation=generation,
                static_gradient=static_gradient,
            )

    def _generate_llm_verbal_gradient(
            self,
            *,
            candidate: Candidate,
            parents: list[Candidate],
            generation: int,
            static_gradient: dict,
    ) -> None:
        prompt_builder = getattr(self.problem, "build_llm_verbal_gradient_prompt", None)
        text_completion = getattr(self.provider, "complete_text", None)
        if not callable(prompt_builder) or not callable(text_completion):
            return
        gradient_config = self.config.evolution.verbal_gradients
        prompt = prompt_builder(
            candidate,
            parents=parents,
            generation=generation,
            static_gradient=static_gradient,
        )
        feedback_record = {
            "type": "verbal_gradient",
            "generation": generation,
            "candidate_id": candidate.id,
            "problem": self.config.problem.type,
            "prompt": prompt,
            "model": getattr(self.provider, "model", None),
            "status": "ok",
        }
        self._llm_gradient_calls_by_generation[generation] = (
            self._llm_gradient_calls_by_generation.get(generation, 0) + 1
        )
        try:
            text = text_completion(prompt, temperature=gradient_config.temperature)
            gradient = parse_llm_verbal_gradient(
                text,
                static_gradient=static_gradient,
                candidate=candidate,
                parents=parents,
                generation=candidate.generation,
            )
            set_candidate_gradient(candidate, gradient)
            feedback_record["response"] = text
            feedback_record["gradient"] = gradient
            self._verbal_gradient_stats["llm_count"] += 1
        except Exception as exc:
            existing_gradient = get_candidate_gradient(candidate) or static_gradient
            existing_gradient = dict(existing_gradient)
            existing_gradient["llm_error"] = _exception_details(exc)
            set_candidate_gradient(candidate, existing_gradient)
            feedback_record["status"] = "error"
            feedback_record["error_details"] = _exception_details(exc)
            self._verbal_gradient_stats["llm_error_count"] += 1
        self.store.save_feedback(feedback_record)
        self.store.save_candidate(candidate)

    def _feedback_context(self, strategy: Strategy, parents: list[Candidate]) -> str:
        gradient_config = self.config.evolution.verbal_gradients
        if not gradient_config.enabled:
            return ""
        return format_parent_verbal_gradients(
            parents,
            strategy=strategy,
            max_chars=gradient_config.max_chars,
        )

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
