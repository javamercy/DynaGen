import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from dynagen.candidates import CandidateStatus, ParsedCandidateResponse
from dynagen.candidates.candidate import Candidate
from dynagen.config import RunConfig
from dynagen.evaluation.base import CandidateEvaluator
from dynagen.evolution.archive import (
    archive_selection_ids,
    CandidateArchive,
    clear_archive_selection,
)
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
            feedback_provider: LLMProvider | None = None,
            search_evaluator: CandidateEvaluator,
            test_evaluator: CandidateEvaluator,
            store: RunStore,
    ) -> None:
        self.config = config
        self.provider = provider
        self.feedback_provider = feedback_provider or provider
        self.search_evaluator = search_evaluator
        self.test_evaluator = test_evaluator
        self.store = store
        self.problem: Problem = problem_for_config(config)
        self.rng = random.Random(config.seed)
        self._candidate_index: dict[str, Candidate] = {}
        self.archive = CandidateArchive(config=config.evolution.archive, problem=config.problem.type)
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
            summary=generation_summary(
                0,
                population.candidates,
                [],
                archive_summary=self._archive_summary(include_entries=False),
            ),
        )
        self._save_archive(0)
        for generation in range(1, self.config.evolution.generations + 1):
            offspring = self._generate_offspring(generation, population)
            next_candidates = select_survivors(population.candidates + offspring, self.config.evolution.population_size)
            population = Population(generation=generation, candidates=next_candidates)
            summary = generation_summary(
                generation,
                population.candidates,
                offspring,
                archive_summary=self._archive_summary(include_entries=False),
            )
            self.store.save_generation(
                generation,
                population=population.candidates,
                offspring=offspring,
                summary=summary,
            )
            self._save_archive(generation)
        search_best = self._search_best(population)
        self.archive.mark_final_selection(
            search_best.id,
            population_ids={candidate.id for candidate in population.candidates},
        )
        test_result = self.test_evaluator.evaluate_code(search_best.code)
        self.store.save_test_result(search_best.id, test_result)
        llm_calls = self._llm_call_summary()
        self.store.save_llm_calls(llm_calls)
        self.store.save_archive_summary(self._archive_summary(include_entries=True))
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
        provider_summaries = self._provider_summaries()
        configured_budget = scheduled_llm_calls(self.config)
        main_summary = provider_summaries[0] if provider_summaries else {}
        feedback_summary = provider_summaries[1] if len(provider_summaries) > 1 else main_summary
        main_model = main_summary.get("llm_model") or getattr(self.provider, "model", None)
        feedback_model = feedback_summary.get("llm_model") or getattr(self.feedback_provider, "model", None)
        candidate_generation_calls = self._sum_provider_metric(provider_summaries, key="candidate_generation_calls") or 0
        feedback_calls = self._sum_provider_metric(provider_summaries, key="feedback_calls") or 0
        total_api_calls = self._sum_provider_metric(provider_summaries, key="total_api_calls") or 0
        failed_calls = self._sum_provider_metric(provider_summaries, key="failed_calls") or 0
        summary = {
            "llm_model": main_model,
            "feedback_llm_model": feedback_model,
            "candidate_generation_calls": candidate_generation_calls,
            "feedback_calls": feedback_calls,
            "reflection_calls": feedback_calls,
            "total_api_calls": total_api_calls,
            "failed_calls": failed_calls,
            "configured_candidate_generation_budget": configured_budget,
            "budget_match": candidate_generation_calls == configured_budget,
            "verbal_gradients": {
                "enabled": self.config.evolution.verbal_gradients.enabled,
                "static_enabled": self.config.evolution.verbal_gradients.static_enabled,
                "llm_enabled": self.config.evolution.verbal_gradients.llm_enabled,
                "llm_every_n_generations": self.config.evolution.verbal_gradients.llm_every_n_generations,
                "llm_model": self.config.evolution.verbal_gradients.llm_model or feedback_model or main_model,
                "feedback_llm_model": feedback_model,
                "temperature": self.config.evolution.verbal_gradients.temperature,
                **self._verbal_gradient_stats,
            },
            "archive": self._archive_summary(include_entries=False),
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
        self._update_archive(candidates, generation=0)
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
        self._update_archive(offspring, generation=generation)
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
        count = parent_count(strategy)
        if not self.archive.enabled or not self.archive.entries:
            parents = select_parents(candidates, count, self.rng)
            clear_archive_selection(parents)
            return parents

        selected: list[Candidate] = []
        selected_ids: set[str] = set()
        archive_min = (
            self.config.evolution.archive.s3_archive_parent_min
            if strategy == Strategy.S3
            else 0
        )
        archive_min = min(archive_min, count)
        if archive_min:
            selected.extend(self.archive.select_parents(
                count=archive_min,
                rng=self.rng,
                candidate_index=self._candidate_index,
                exclude_ids=selected_ids,
                diversify_buckets=True,
            ))
            selected_ids.update(candidate.id for candidate in selected)

        while len(selected) < count:
            remaining = count - len(selected)
            use_archive = self.rng.random() < self.config.evolution.archive.parent_sample_probability
            next_parent: list[Candidate] = []
            if use_archive:
                next_parent = self.archive.select_parents(
                    count=1,
                    rng=self.rng,
                    candidate_index=self._candidate_index,
                    exclude_ids=selected_ids,
                    diversify_buckets=strategy == Strategy.S3,
                )
            if not next_parent:
                pool = [candidate for candidate in candidates if candidate.id not in selected_ids]
                if pool:
                    next_parent = select_parents(pool, min(1, remaining), self.rng)
                    clear_archive_selection(next_parent)
            if not next_parent and not use_archive:
                next_parent = self.archive.select_parents(
                    count=1,
                    rng=self.rng,
                    candidate_index=self._candidate_index,
                    exclude_ids=selected_ids,
                    diversify_buckets=strategy == Strategy.S3,
                )
            if not next_parent:
                break
            selected.extend(next_parent)
            selected_ids.update(candidate.id for candidate in next_parent)

        if not selected:
            selected = select_parents(candidates, count, self.rng)
            clear_archive_selection(selected)
        if archive_selection_ids(selected):
            self.archive.note_offspring_with_archive_parent()
        return selected

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
        if (
            not gradient_config.enabled
            or not gradient_config.llm_enabled
            or generation % gradient_config.llm_every_n_generations != 0
        ):
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
        text_completion = getattr(self.feedback_provider, "complete_text", None)
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
            "model": getattr(self.feedback_provider, "model", None),
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
                generation=generation,
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

    def _provider_summary(self, provider: LLMProvider | None) -> dict[str, object]:
        if provider is None:
            return {}
        summary_getter = getattr(provider, "summary", None)
        summary = dict(summary_getter()) if callable(summary_getter) else {}
        model = summary.get("llm_model") or getattr(provider, "model", None)
        if model is not None:
            summary["llm_model"] = model
        return summary

    def _sum_provider_metric(
            self,
            summaries: list[dict[str, object]],
            *,
            key: str,
    ) -> int | None:
        values: list[int] = []
        for summary in summaries:
            value = summary.get(key)
            if value is None:
                continue
            try:
                values.append(int(value))
            except (TypeError, ValueError):
                continue
        return None if not values else sum(values)

    def _feedback_context(self, strategy: Strategy, parents: list[Candidate]) -> str:
        gradient_config = self.config.evolution.verbal_gradients
        if not gradient_config.enabled:
            return ""
        return format_parent_verbal_gradients(
            parents,
            strategy=strategy,
            max_chars=gradient_config.max_chars,
        )

    def _provider_summaries(self) -> list[dict[str, object]]:
        providers: list[LLMProvider] = [self.provider]
        if self.feedback_provider is not self.provider:
            providers.append(self.feedback_provider)
        summaries = [self._provider_summary(provider) for provider in providers]
        return [summary for summary in summaries if summary]

    def _update_archive(self, candidates: list[Candidate], *, generation: int) -> None:
        if not self.archive.enabled:
            return
        profile_builder = getattr(self.problem, "build_archive_profile", None)
        if not callable(profile_builder):
            return
        self.archive.update(
            candidates,
            generation=generation,
            profile_builder=profile_builder,
        )
        for candidate in candidates:
            self.store.save_candidate(candidate)

    def _save_archive(self, generation: int) -> None:
        if self.archive.enabled:
            self.store.save_archive(generation, self._archive_summary(include_entries=True))

    def _archive_summary(self, *, include_entries: bool) -> dict:
        return self.archive.summary(include_entries=include_entries)

    def _search_best(self, population: Population) -> Candidate:
        if (
            not self.archive.enabled
            or not self.config.evolution.archive.final_selection_uses_archive
        ):
            return population.best
        candidates_by_id = {candidate.id: candidate for candidate in population.candidates}
        for candidate in self.archive.candidates(self._candidate_index):
            candidates_by_id.setdefault(candidate.id, candidate)
        return select_survivors(list(candidates_by_id.values()), 1)[0]

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
