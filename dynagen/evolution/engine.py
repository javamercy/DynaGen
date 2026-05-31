import logging
import math
import os
import random
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from dynagen.candidates import CandidateStatus, ParsedCandidateResponse
from dynagen.candidates.candidate import (
    Candidate,
    MAXIMIZED_SCORE_NAMES,
    MINIMIZED_SCORE_NAMES,
    NAMED_SCORE_NAMES,
)
from dynagen.config import RunConfig
from dynagen.evaluation.base import CandidateEvaluator
from dynagen.evolution.history import (
    history_selection_ids,
    CandidateHistory,
    clear_history_selection,
)
from dynagen.evolution.population import Population
from dynagen.evolution.selection import select_parents, select_survivors
from dynagen.evolution.strategies import parent_count, Strategy
from dynagen.evolution.verbal_gradient import (
    candidate_has_llm_gradient,
    format_parent_verbal_gradients,
    normalize_verbal_gradient,
    parse_llm_verbal_gradient,
    set_candidate_gradient,
)
from dynagen.evolution.committee import (
    assign_instances,
    niche_probabilities,
    select_committee,
)
from dynagen.llm.base import LLMBudgetExceeded, LLMProvider
from dynagen.persistence.run_store import RunStore
from dynagen.problems import problem_for_config
from dynagen.problems.base import Problem
from dynagen.reporting.summary import (
    build_committee_final_report,
    build_final_report,
    generation_summary,
)

logger = logging.getLogger(__name__)


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
        self.history = CandidateHistory(
            config=config.evolution.history, problem=config.problem.type
        )
        self._llm_gradient_calls_by_generation: dict[int, int] = {}
        self._verbal_gradient_stats: dict[str, int] = {
            "llm_count": 0,
            "llm_error_count": 0,
        }
        self._committee_specialists: list[Candidate] = []
        self._committee_assignments: dict[str, list[str]] = {}
        self._committee_niche_iteration: dict[str, int] = {}
        self._exploration_burst_functions: list[str] = []

    def run(self) -> Population:
        output_mode = self.config.evolution.output_mode
        population = self._initial_population()
        if output_mode != "single":
            self._recompute_committee()

        self.store.save_generation(
            0,
            population=population.candidates,
            offspring=[],
            summary=generation_summary(
                0,
                population.candidates,
                [],
                history_summary=self._history_summary(include_entries=False),
            ),
        )
        self._save_history(0)
        for generation in range(1, self.config.evolution.generations + 1):
            try:
                offspring = self._generate_offspring(generation, population)
            except LLMBudgetExceeded:
                break
            if output_mode != "single":
                pool = list(population.candidates) + offspring
                pool_ids = {c.id for c in pool}
                for c in self.history.candidates(self._candidate_index):
                    if c.id not in pool_ids and c.status != CandidateStatus.ERROR:
                        pool.append(c)
                        pool_ids.add(c.id)
                next_candidates = self._committee_survivors(
                    pool, self.config.evolution.population_size
                )
            else:
                next_candidates = select_survivors(
                    population.candidates + offspring,
                    self.config.evolution.population_size,
                )
            population = Population(generation=generation, candidates=next_candidates)
            summary = generation_summary(
                generation,
                population.candidates,
                offspring,
                history_summary=self._history_summary(include_entries=False),
            )
            self.store.save_generation(
                generation,
                population=population.candidates,
                offspring=offspring,
                summary=summary,
            )
            self._save_history(generation)
            if (
                output_mode != "single"
                and generation % self.config.evolution.niche.cadence_generations == 0
            ):
                self._recompute_committee()

        if output_mode == "single":
            return self._finalize_single(population)
        return self._finalize_committee(population)

    def _finalize_single(self, population: Population) -> Population:
        search_best = self._search_best(population)
        self.history.mark_final_selection(
            search_best.id,
            population_ids={candidate.id for candidate in population.candidates},
        )
        problem_tag = self._problem_tag()
        logger.info("[%s] testing best candidate %s", problem_tag, search_best.id)
        test_result = self.test_evaluator.evaluate_code(search_best.code)
        logger.info(
            "[%s] test complete | candidate=%s | status=%s | score=%s",
            problem_tag,
            search_best.id,
            test_result.status,
            test_result.score,
        )
        self.store.save_test_result(search_best.id, test_result)
        llm_calls = self._llm_call_summary()
        self.store.save_llm_calls(llm_calls)
        self.store.save_history_summary(self._history_summary(include_entries=True))
        self.store.write_final_report(
            build_final_report(
                population.candidates,
                search_best=search_best,
                test_result=test_result,
                llm_calls=llm_calls,
            )
        )
        return population

    def _finalize_committee(self, population: Population) -> Population:
        self._recompute_committee()
        specialists = self._committee_specialists
        if not specialists:
            return self._finalize_single(population)

        problem_tag = self._problem_tag()
        search_best = self._search_best(population)

        test_results: dict[str, object] = {}
        for specialist in specialists:
            logger.info("[%s] testing specialist %s", problem_tag, specialist.id)
            result = self.test_evaluator.evaluate_code(specialist.code)
            test_results[specialist.id] = {
                "candidate_id": specialist.id,
                "name": specialist.name,
                "status": result.status,
                "score": result.score,
                "score_name": result.score_name,
                "metrics": result.metrics,
            }
            logger.info(
                "[%s] test complete | specialist=%s | status=%s | score=%s",
                problem_tag,
                specialist.id,
                result.status,
                result.score,
            )

        self.store.save_committee_results(
            specialists=[c.id for c in specialists],
            assignments=self._committee_assignments,
            test_results=test_results,
        )

        llm_calls = self._llm_call_summary()
        self.store.save_llm_calls(llm_calls)
        self.store.save_history_summary(self._history_summary(include_entries=True))

        vbs_scores: dict[str, float] = {}
        if specialists:
            all_instances: list[str] = []
            test_per_function: dict[str, dict[str, float]] = {}
            for sp in specialists:
                tr = test_results.get(sp.id)
                af = {}
                if isinstance(tr, dict):
                    metrics = tr.get("metrics") or {}
                    af = (
                        metrics.get("aocc_by_function")
                        or metrics.get("score_by_instance_size")
                        or {}
                    )
                if isinstance(af, dict):
                    test_per_function[sp.id] = {str(k): float(v) for k, v in af.items()}
                    all_instances.extend(
                        k for k in test_per_function[sp.id] if k not in all_instances
                    )
            for instance in all_instances:
                vbs_scores[instance] = max(
                    (test_per_function.get(sp.id, {}).get(instance, 0.0))
                    for sp in specialists
                )

        self.store.write_final_report(
            build_committee_final_report(
                population.candidates,
                search_best=search_best,
                specialists=specialists,
                assignments=self._committee_assignments,
                test_results=test_results,
                vbs_scores=vbs_scores,
                llm_calls=llm_calls,
            )
        )
        return population

    def _llm_call_summary(self) -> dict:
        provider_summaries = self._provider_summaries()
        configured_budget = self.config.llm.llm_call_budget or scheduled_llm_calls(
            self.config
        )
        main_summary = provider_summaries[0] if provider_summaries else {}
        feedback_summary = (
            provider_summaries[1] if len(provider_summaries) > 1 else main_summary
        )
        main_model = main_summary.get("llm_model") or getattr(
            self.provider, "model", None
        )
        feedback_model = feedback_summary.get("llm_model") or getattr(
            self.feedback_provider, "model", None
        )
        candidate_generation_calls = (
            self._sum_provider_metric(
                provider_summaries, key="candidate_generation_calls"
            )
            or 0
        )
        feedback_calls = (
            self._sum_provider_metric(provider_summaries, key="feedback_calls") or 0
        )
        total_api_calls = (
            self._sum_provider_metric(provider_summaries, key="total_api_calls") or 0
        )
        failed_calls = (
            self._sum_provider_metric(provider_summaries, key="failed_calls") or 0
        )
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
                "llm_every_n_generations": self.config.evolution.verbal_gradients.llm_every_n_generations,
                "llm_model": self.config.evolution.verbal_gradients.llm_model
                or feedback_model
                or main_model,
                "feedback_llm_model": feedback_model,
                "temperature": self.config.evolution.verbal_gradients.temperature,
                **self._verbal_gradient_stats,
            },
            "history": self._history_summary(include_entries=False),
        }
        return summary

    # ------------------------------------------------------------------
    # Initial population: parallel LLM calls + parallel evaluations
    # ------------------------------------------------------------------

    def _initial_population(self) -> Population:
        logger.info(
            "[%s] initializing population | size=%d",
            self._problem_tag(),
            self.config.evolution.population_size,
        )
        roles = self.problem.initial_roles(self.config.evolution.population_size)
        tasks = self._build_initial_tasks(roles)
        candidates = self._execute_tasks_parallel(tasks)
        self._register_candidates(candidates)
        self._update_history(candidates, generation=0)
        return Population.from_candidates(
            0, candidates, size=self.config.evolution.population_size
        )

    def _build_initial_tasks(self, roles: list) -> list[_CandidateTask]:
        tasks: list[_CandidateTask] = []
        for role in roles:
            messages = self.problem.build_initial_prompt(role)
            prompt = _format_messages(messages)
            candidate_id = self.store.next_candidate_id()
            tasks.append(
                _CandidateTask(
                    candidate_id=candidate_id,
                    generation=0,
                    strategy=f"initial:{role.slot}",
                    parents=[],
                    messages=messages,
                    prompt=prompt,
                )
            )
        return tasks

    # ------------------------------------------------------------------
    # Offspring generation: parallel LLM calls + parallel evaluations
    # ------------------------------------------------------------------

    def _generate_offspring(
        self, generation: int, population: Population
    ) -> list[Candidate]:
        if self.config.evolution.output_mode == "single":
            return self._generate_single_offspring(generation, population)
        return self._generate_committee_offspring(generation, population)

    def _generate_single_offspring(
        self, generation: int, population: Population
    ) -> list[Candidate]:
        tasks = self._build_offspring_tasks(generation, population)
        offspring = self._execute_tasks_parallel(tasks)
        self._register_candidates(offspring)
        self._update_history(offspring, generation=generation)
        return offspring

    def _generate_committee_offspring(
        self, generation: int, population: Population
    ) -> list[Candidate]:
        niche_config = self.config.evolution.niche
        cadence = niche_config.cadence_generations
        problem_tag = self._problem_tag()

        if generation % cadence == 0 or not self._committee_specialists:
            self._recompute_committee()
            specialists = self._committee_specialists
            if not specialists:
                return self._generate_single_offspring(generation, population)

            logger.info(
                "[%s] cadence generation %d — all %d niches",
                problem_tag,
                generation,
                len(specialists),
            )
            all_offspring: list[Candidate] = []
            if self._exploration_burst_functions:
                all_offspring += self._burst_exploration(generation, population)
                self._exploration_burst_functions.clear()
            for specialist in specialists:
                assigned = self._committee_assignments.get(specialist.id, [])
                scores = self.problem.per_instance_scores(specialist)
                niche_mean = (
                    sum(scores.get(k, 0.0) for k in assigned) / len(assigned)
                    if assigned
                    else 0.0
                )
                logger.info(
                    "[%s]   niche gen | specialist=%s | functions=%s | niche_mean=%.4f",
                    problem_tag,
                    specialist.id,
                    ",".join(assigned[:5]) + ("..." if len(assigned) > 5 else ""),
                    niche_mean,
                )
                tasks = self._build_niche_offspring_tasks(
                    generation, population, specialist
                )
                if tasks:
                    offspring = self._execute_tasks_parallel(tasks)
                    self._register_candidates(offspring)
                    self._update_history(offspring, generation=generation)
                    all_offspring.extend(offspring)
            return all_offspring

        specialists = self._committee_specialists
        if not specialists:
            return self._generate_single_offspring(generation, population)

        chosen = self._select_niche(specialists)
        if chosen is None:
            return self._generate_single_offspring(generation, population)

        assigned = self._committee_assignments.get(chosen.id, [])
        scores = self.problem.per_instance_scores(chosen)
        niche_mean = (
            sum(scores.get(k, 0.0) for k in assigned) / len(assigned)
            if assigned
            else 0.0
        )
        logger.info(
            "[%s] niche gen %d | specialist=%s | functions=%s | niche_mean=%.4f",
            problem_tag,
            generation,
            chosen.id,
            ",".join(assigned[:5]) + ("..." if len(assigned) > 5 else ""),
            niche_mean,
        )

        tasks = self._build_niche_offspring_tasks(generation, population, chosen)
        if not tasks:
            return []

        offspring = self._execute_tasks_parallel(tasks)
        self._register_candidates(offspring)
        self._update_history(offspring, generation=generation)

        if (
            not self.config.evolution.niche_population_mix
            and self.config.evolution.archive_niche_replacement
        ):
            self._replace_if_better(chosen, assigned, offspring, population)

        if self._exploration_burst_functions and generation == 1:
            offspring += self._burst_exploration(generation, population)
            self._exploration_burst_functions.clear()

        return offspring

    def _recompute_committee(self) -> None:
        pool = list(self._candidate_index.values())
        history_candidates = self.history.candidates(self._candidate_index)
        for c in history_candidates:
            if c.id not in self._candidate_index:
                pool.append(c)
                self._candidate_index[c.id] = c

        per_instance_scores_fn = lambda c: self.problem.per_instance_scores(c)
        specialists, assignments = select_committee(
            pool,
            per_instance_scores_fn=per_instance_scores_fn,
            committee_size=self.config.evolution.committee_size,
            method="kmeans",
        )
        self._committee_specialists = specialists
        self._committee_assignments = assignments

        if len(specialists) < self.config.evolution.committee_size:
            all_assigned = set()
            for inst_list in assignments.values():
                all_assigned.update(inst_list)
            all_instances = set()
            for c in pool:
                all_instances.update(per_instance_scores_fn(c).keys())
            self._exploration_burst_functions = sorted(all_instances - all_assigned)

        problem_tag = self._problem_tag()
        all_instances_count = len(
            set(k for c in pool for k in per_instance_scores_fn(c).keys())
        )
        logger.info(
            "[%s] committee recomputed | specialists=%d | total_instances=%d",
            problem_tag,
            len(specialists),
            all_instances_count,
        )
        for sp in specialists:
            assigned = assignments.get(sp.id, [])
            scores = per_instance_scores_fn(sp)
            niche_mean = (
                sum(scores.get(k, 0.0) for k in assigned) / len(assigned)
                if assigned
                else 0.0
            )
            niche_contrib = niche_mean * len(assigned) / max(1, all_instances_count)
            global_mean = (
                sp.metrics.get("mean_aocc", sp.metrics.get("mean_gap", 0.0))
                if isinstance(sp.metrics, dict)
                else 0.0
            )
            logger.info(
                "[%s]   %s: niche=[%s](%d) niche_mean=%.4f contrib=%.4f global_mean=%.4f",
                problem_tag,
                sp.id,
                ",".join(assigned[:5]) + ("..." if len(assigned) > 5 else ""),
                len(assigned),
                niche_mean,
                niche_contrib,
                global_mean,
            )

    def _replace_if_better(
        self,
        specialist: Candidate,
        assigned: list[str],
        offspring: list[Candidate],
        population: Population,
    ) -> None:
        if not assigned:
            return
        per_instance_scores_fn = lambda c: self.problem.per_instance_scores(c)
        current_scores = per_instance_scores_fn(specialist)
        current_mean = sum(current_scores.get(k, 0.0) for k in assigned) / len(assigned)
        for c in offspring:
            if c.status == CandidateStatus.ERROR:
                continue
            scores = per_instance_scores_fn(c)
            new_mean = sum(scores.get(k, 0.0) for k in assigned) / len(assigned)
            if new_mean > current_mean:
                for i, sp in enumerate(self._committee_specialists):
                    if sp.id == specialist.id:
                        self._committee_specialists[i] = c
                        self._candidate_index[c.id] = c
                        problem_tag = self._problem_tag()
                        logger.info(
                            "[%s] archive replaced | %s → %s | niche_mean %.4f → %.4f",
                            problem_tag,
                            specialist.id,
                            c.id,
                            current_mean,
                            new_mean,
                        )
                        break

    def _burst_exploration(
        self, generation: int, population: Population
    ) -> list[Candidate]:
        problem_tag = self._problem_tag()
        orphaned = self._exploration_burst_functions
        logger.info(
            "[%s] exploration burst | gen=%d | functions=%s",
            problem_tag,
            generation,
            ",".join(orphaned[:8]) + ("..." if len(orphaned) > 8 else ""),
        )
        temp_specialist = Candidate(
            id="_burst_",
            generation=generation,
            strategy="exploration_burst",
            name="_burst_",
            metrics={"aocc_by_function": {k: 0.0 for k in orphaned}},
        )
        tasks: list[_CandidateTask] = []
        burst_strategies = self.config.evolution.committee_recovery_strategies
        for strategy in burst_strategies:
            for _ in range(self.config.evolution.offspring_per_strategy):
                candidate_id = self.store.next_candidate_id()
                parents = self._parents_for_niche(orphaned, population)
                strategy_parents = self._select_strategy_parents(strategy, parents)
                messages = self.problem.build_evolution_prompt(
                    strategy,
                    strategy_parents[:1],
                    feedback_context="",
                )
                tasks.append(
                    _CandidateTask(
                        candidate_id=candidate_id,
                        generation=generation,
                        strategy=str(strategy),
                        parents=strategy_parents,
                        messages=messages,
                        prompt=_format_messages(messages),
                    )
                )
        offspring = self._execute_tasks_parallel(tasks)
        self._register_candidates(offspring)
        self._update_history(offspring, generation=generation)
        return offspring

    def _committee_survivors(
        self, pool: list[Candidate], population_size: int
    ) -> list[Candidate]:
        if (
            not self.config.evolution.niche_population_mix
            or not self._committee_specialists
        ):
            return select_survivors(pool, population_size)

        global_best = list(select_survivors(pool, population_size))

        per_instance_scores_fn = lambda c: self.problem.per_instance_scores(c)
        niche_best: dict[str, Candidate] = {}
        for sp in self._committee_specialists:
            assigned = self._committee_assignments.get(sp.id, [])
            if not assigned:
                continue
            best = None
            best_niche_mean = -1.0
            for c in pool:
                if c.status == CandidateStatus.ERROR:
                    continue
                scores = per_instance_scores_fn(c)
                relevant = [scores.get(k, 0.0) for k in assigned if k in scores]
                if not relevant:
                    continue
                niche_mean = sum(relevant) / len(relevant)
                if niche_mean > best_niche_mean:
                    best_niche_mean = niche_mean
                    best = c
            if best is not None:
                niche_best[sp.id] = best

        result: list[Candidate] = []
        picked_niches: set[str] = set()
        global_idx = 0

        while len(result) < population_size:
            unpicked = [
                sp
                for sp in self._committee_specialists
                if sp.id in niche_best and sp.id not in picked_niches
            ]
            if unpicked:
                sp = unpicked[0]
                result.append(niche_best[sp.id])
                picked_niches.add(sp.id)
            elif global_idx < len(global_best):
                result.append(global_best[global_idx])
                global_idx += 1
            else:
                break

        return result

    def _select_niche(self, specialists: list[Candidate]) -> Candidate | None:
        if not specialists:
            return None

        per_instance_scores_fn = lambda c: self.problem.per_instance_scores(c)
        probs = niche_probabilities(
            specialists,
            instance_assignments=self._committee_assignments,
            per_instance_scores_fn=per_instance_scores_fn,
            improvement_weight=self.config.evolution.niche.improvement_weight,
            improvement_power=self.config.evolution.niche.improvement_power,
        )

        roll = self.rng.random()
        cumulative = 0.0
        for sp in specialists:
            cumulative += probs.get(sp.id, 0.0)
            if roll < cumulative or cumulative >= 1.0:
                return sp
        return specialists[-1] if specialists else None

    def _build_niche_offspring_tasks(
        self,
        generation: int,
        population: Population,
        specialist: Candidate,
    ) -> list[_CandidateTask]:
        assigned = self._committee_assignments.get(specialist.id, [])
        niche_parents = self._parents_for_niche(assigned, population)

        if not niche_parents:
            niche_parents = select_parents(
                population.candidates, min(3, len(population.candidates)), self.rng
            )
            if niche_parents:
                clear_history_selection(niche_parents)

        tasks: list[_CandidateTask] = []
        available_strategies = self._effective_strategies(generation=generation)
        for strategy in available_strategies:
            for _ in range(self.config.evolution.offspring_per_strategy):
                candidate_id = self.store.next_candidate_id()
                strategy_parents = self._select_strategy_parents(
                    strategy, niche_parents
                )
                self._ensure_parent_verbal_gradients(strategy_parents, generation)
                feedback_context = self._feedback_context(strategy_parents)
                prompt_parents = (
                    strategy_parents[:1] if feedback_context else strategy_parents
                )
                messages = self.problem.build_evolution_prompt(
                    strategy,
                    prompt_parents,
                    feedback_context=feedback_context,
                )
                prompt = _format_messages(messages)
                tasks.append(
                    _CandidateTask(
                        candidate_id=candidate_id,
                        generation=generation,
                        strategy=strategy,
                        parents=strategy_parents,
                        messages=messages,
                        prompt=prompt,
                    )
                )
        return tasks

    def _parents_for_niche(
        self,
        assigned_instances: list[str],
        population: Population,
    ) -> list[Candidate]:
        if not assigned_instances:
            return list(population.candidates)

        per_instance_scores_fn = lambda c: self.problem.per_instance_scores(c)
        scored: list[tuple[Candidate, float]] = []
        for c in population.candidates:
            if c.status == CandidateStatus.ERROR:
                continue
            scores = per_instance_scores_fn(c)
            if not scores:
                continue
            relevant = [scores.get(k, 0.0) for k in assigned_instances if k in scores]
            if not relevant:
                continue
            scored.append((c, sum(relevant) / len(relevant)))

        history_entries = list(self.history.entries.values())
        for entry in sorted(history_entries, key=lambda e: -e.history_score):
            c = self._candidate_index.get(entry.candidate_id)
            if c is None or c in population.candidates:
                continue
            scores = per_instance_scores_fn(c)
            if not scores:
                continue
            relevant = [scores.get(k, 0.0) for k in assigned_instances if k in scores]
            if not relevant:
                continue
            scored.append((c, sum(relevant) / len(relevant)))

        scored.sort(key=lambda item: -item[1])
        niche_pool = [
            c for c, _ in scored[: max(5, self.config.evolution.population_size)]
        ]
        if not niche_pool:
            return list(population.candidates)

        return niche_pool

    def _effective_strategies(self, *, generation: int = 0) -> list[Strategy]:
        threshold = self.config.evolution.archive_mode_strategy_weights_after_generation
        archive_active = (
            self.config.evolution.archive_mode_strategy_weights is not None
            and generation >= threshold
        )
        weights = (
            self.config.evolution.archive_mode_strategy_weights
            if archive_active
            else self.config.evolution.strategy_weights
        )
        if not weights:
            return self.config.evolution.strategies
        active = []
        for strategy in self.config.evolution.strategies:
            name = strategy.value if hasattr(strategy, "value") else str(strategy)
            weight = weights.get(name, 0.0)
            if self.rng.random() < weight:
                active.append(strategy)
        if not active:
            active = [Strategy.M1_COMPONENT_REPLACEMENT]
        return active

    def _build_offspring_tasks(
        self,
        generation: int,
        population: Population,
    ) -> list[_CandidateTask]:
        tasks: list[_CandidateTask] = []
        available_strategies = self._effective_strategies(generation=generation)
        for strategy in available_strategies:
            for _ in range(self.config.evolution.offspring_per_strategy):
                candidate_id = self.store.next_candidate_id()
                parents = self._select_strategy_parents(strategy, population.candidates)
                self._ensure_parent_verbal_gradients(parents, generation)
                feedback_context = self._feedback_context(parents)
                prompt_parents = parents[:1] if feedback_context else parents
                messages = self.problem.build_evolution_prompt(
                    strategy,
                    prompt_parents,
                    feedback_context=feedback_context,
                )
                prompt = _format_messages(messages)
                tasks.append(
                    _CandidateTask(
                        candidate_id=candidate_id,
                        generation=generation,
                        strategy=strategy,
                        parents=parents,
                        messages=messages,
                        prompt=prompt,
                    )
                )
        return tasks

    # ------------------------------------------------------------------
    # Parallel task execution: LLM call → evaluation → persist
    # ------------------------------------------------------------------

    def _execute_tasks_parallel(self, tasks: list[_CandidateTask]) -> list[Candidate]:
        """Execute LLM calls and evaluations concurrently, preserving task order."""
        if not tasks:
            return []

        max_workers = min(len(tasks), os.cpu_count() or 1)
        results: list[Candidate] = [None] * len(tasks)  # type: ignore[list-item]
        budget_exceeded = False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_single_task, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except LLMBudgetExceeded:
                    budget_exceeded = True
                    for f in future_to_index:
                        if not f.done():
                            f.cancel()
                except CancelledError:
                    pass

        if budget_exceeded:
            raise LLMBudgetExceeded(
                "LLM call budget exceeded during candidate generation"
            )
        return results

    def _process_single_task(self, task: _CandidateTask) -> Candidate:
        """Run LLM call + evaluation for a single candidate. Always returns a Candidate."""
        candidate: Candidate | None = None
        try:
            problem_tag = self._problem_tag()
            logger.info(
                "[%s] llm call | candidate=%s | generation=%d | strategy=%s",
                problem_tag,
                task.candidate_id,
                task.generation,
                task.strategy,
            )
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
            logger.info(
                "[%s] evaluation | candidate=%s | status=%s | score=%s",
                problem_tag,
                candidate.id,
                candidate.status,
                candidate.score_value,
            )
        except LLMBudgetExceeded:
            raise
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

    def _select_strategy_parents(
        self, strategy: Strategy, candidates: list[Candidate]
    ) -> list[Candidate]:
        count = parent_count(strategy)
        if not self.history.enabled or not self.history.entries:
            parents = select_parents(candidates, count, self.rng)
            clear_history_selection(parents)
            return parents

        selected: list[Candidate] = []
        selected_ids: set[str] = set()
        history_min = 0
        history_min = min(history_min, count)
        if history_min:
            selected.extend(
                self.history.select_parents(
                    count=history_min,
                    rng=self.rng,
                    candidate_index=self._candidate_index,
                    exclude_ids=selected_ids,
                    diversify_buckets=True,
                )
            )
            selected_ids.update(candidate.id for candidate in selected)

        while len(selected) < count:
            remaining = count - len(selected)
            use_history = (
                self.rng.random()
                < self.config.evolution.history.parent_sample_probability
            )
            next_parent: list[Candidate] = []
            if use_history:
                next_parent = self.history.select_parents(
                    count=1,
                    rng=self.rng,
                    candidate_index=self._candidate_index,
                    exclude_ids=selected_ids,
                    diversify_buckets=False,
                )
            if not next_parent:
                pool = [
                    candidate
                    for candidate in candidates
                    if candidate.id not in selected_ids
                ]
                if pool:
                    next_parent = select_parents(pool, min(1, remaining), self.rng)
                    clear_history_selection(next_parent)
            if not next_parent and not use_history:
                next_parent = self.history.select_parents(
                    count=1,
                    rng=self.rng,
                    candidate_index=self._candidate_index,
                    exclude_ids=selected_ids,
                    diversify_buckets=False,
                )
            if not next_parent:
                break
            selected.extend(next_parent)
            selected_ids.update(candidate.id for candidate in next_parent)

        if not selected:
            selected = select_parents(candidates, count, self.rng)
            clear_history_selection(selected)
        if history_selection_ids(selected):
            self.history.note_offspring_with_history_parent()
        return selected

    def _empty_metrics(self) -> dict:
        metrics_getter = getattr(self.search_evaluator, "empty_metrics", None)
        return dict(metrics_getter()) if callable(metrics_getter) else {}

    def _ensure_parent_verbal_gradients(
        self,
        parents: list[Candidate],
        generation: int,
    ) -> None:
        gradient_config = self.config.evolution.verbal_gradients
        if (
            not gradient_config.enabled
            or generation % gradient_config.llm_every_n_generations != 0
        ):
            return
        problem_tag = self._problem_tag()
        for parent in parents:
            parent_parents = self._resolve_parents(parent)
            if candidate_has_llm_gradient(parent):
                continue
            if (
                self._llm_gradient_calls_by_generation.get(generation, 0)
                >= gradient_config.max_llm_calls_per_generation
            ):
                continue
            logger.info(
                "[%s] reflection llm call | parent=%s | generation=%d",
                problem_tag,
                parent.id,
                generation,
            )
            self._generate_llm_verbal_gradient(
                candidate=parent,
                parents=parent_parents,
                generation=generation,
            )

    def _generate_llm_verbal_gradient(
        self,
        *,
        candidate: Candidate,
        parents: list[Candidate],
        generation: int,
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
        )
        feedback_record = {
            "type": "reflection",
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
                problem=self.config.problem.type,
                candidate=candidate,
                parents=parents,
                generation=generation,
            )
            set_candidate_gradient(candidate, gradient)
            feedback_record["response"] = text
            feedback_record["gradient"] = gradient
            self._verbal_gradient_stats["llm_count"] += 1
        except Exception as exc:
            error_gradient = normalize_verbal_gradient(
                {},
                fallback_problem=self.config.problem.type,
                fallback_candidate=candidate,
                fallback_generation=generation,
                fallback_parents=parents,
                source="llm_error",
            )
            error_gradient["llm_error"] = _exception_details(exc)
            set_candidate_gradient(candidate, error_gradient)
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

    def _feedback_context(self, parents: list[Candidate]) -> str:
        gradient_config = self.config.evolution.verbal_gradients
        if not gradient_config.enabled:
            return ""
        fields = _reflection_fields_for_strategy(None)
        return format_parent_verbal_gradients(
            parents,
            fields=fields,
        )

    def _provider_summaries(self) -> list[dict[str, object]]:
        providers: list[LLMProvider] = [self.provider]
        if self.feedback_provider is not self.provider:
            providers.append(self.feedback_provider)
        summaries = [self._provider_summary(provider) for provider in providers]
        return [summary for summary in summaries if summary]

    def _update_history(self, candidates: list[Candidate], *, generation: int) -> None:
        if not self.history.enabled:
            return
        profile_builder = getattr(self.problem, "build_history_profile", None)
        if not callable(profile_builder):
            return
        self.history.update(
            candidates,
            generation=generation,
            profile_builder=profile_builder,
        )
        for candidate in candidates:
            self.store.save_candidate(candidate)

    def _save_history(self, generation: int) -> None:
        if self.history.enabled:
            self.store.save_history(
                generation, self._history_summary(include_entries=True)
            )

    def _history_summary(self, *, include_entries: bool) -> dict:
        return self.history.summary(include_entries=include_entries)

    def _search_best(self, population: Population) -> Candidate:
        if (
            not self.history.enabled
            or not self.config.evolution.history.final_selection_uses_history
        ):
            return population.best
        candidates_by_id = {
            candidate.id: candidate for candidate in population.candidates
        }
        for candidate in self.history.candidates(self._candidate_index):
            candidates_by_id.setdefault(candidate.id, candidate)
        candidates = [
            c for c in candidates_by_id.values() if c.status != CandidateStatus.TIMEOUT
        ]
        if not candidates:
            candidates = list(candidates_by_id.values())
        return select_survivors(candidates, 1)[0]

    def _resolve_parents(self, candidate: Candidate) -> list[Candidate]:
        return [
            self._candidate_index[parent_id]
            for parent_id in candidate.parents
            if parent_id in self._candidate_index
        ]

    def _register_candidates(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            self._candidate_index[candidate.id] = candidate

    def _problem_tag(self) -> str:
        return self.config.problem.type.upper()


def _reflection_fields_for_strategy(strategy: str | None) -> set[str]:
    if str(strategy) in ("S2", "m1_component_replacement"):
        return {"summary", "aim", "preserve", "change", "avoid"}
    return {"summary", "aim"}


def scheduled_llm_calls(config: RunConfig) -> int:
    weights = (
        config.evolution.archive_mode_strategy_weights
        or config.evolution.strategy_weights
    )
    if weights:
        expected_strategies = sum(weights.values())
    else:
        expected_strategies = len(config.evolution.strategies)
    expected_calls = int(
        config.evolution.generations
        * expected_strategies
        * config.evolution.offspring_per_strategy
    )
    return config.evolution.population_size + max(0, expected_calls)


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
    score_name = _metric_score_name(candidate_metrics)
    if score_name is not None:
        score_value = _default_score_value(score_name)
        candidate_metrics["score_name"] = score_name
        candidate_metrics[score_name] = score_value
    return Candidate(
        id=candidate_id,
        generation=generation,
        strategy=strategy,
        name=response.name,
        thought=response.thought,
        code=response.code,
        parents=list(parents or []),
        fitness=None,
        distance=score_value if score_name in MINIMIZED_SCORE_NAMES else None,
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
    score_name = _metric_score_name(candidate_metrics)
    if score_name is not None:
        score_value = _default_score_value(score_name)
        candidate_metrics["score_name"] = score_name
        candidate_metrics[score_name] = score_value
    return Candidate(
        id=candidate_id,
        generation=generation,
        strategy=strategy,
        name="",
        thought="",
        code="",
        parents=list(parents or []),
        fitness=None if score_name is not None else math.inf,
        distance=score_value if score_name in MINIMIZED_SCORE_NAMES else None,
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
    score_name = candidate.score_name
    if score_name != "fitness":
        score_value = _default_score_value(score_name)
        candidate.distance = (
            score_value if score_name in MINIMIZED_SCORE_NAMES else None
        )
        candidate.fitness = None
        candidate.metrics["score_name"] = score_name
        candidate.metrics[score_name] = score_value
    else:
        candidate.fitness = math.inf
    candidate.error_details = error_details


def _uses_distance_metrics(metrics: dict) -> bool:
    return _metric_score_name(metrics) in MINIMIZED_SCORE_NAMES


def _metric_score_name(metrics: dict) -> str | None:
    if metrics.get("problem") == "bbob":
        return "mean_aocc"
    if metrics.get("problem") == "dvrp":
        return "ttt"
    if metrics.get("problem") == "vrp":
        return "gap"
    if metrics.get("problem") == "tsp":
        return "distance"
    score_name = metrics.get("score_name")
    if score_name in NAMED_SCORE_NAMES:
        return str(score_name)
    if "ttt" in metrics:
        return "ttt"
    if "distance" in metrics:
        return "distance"
    if "mean_aocc" in metrics:
        return "mean_aocc"
    return None


def _default_score_value(score_name: str) -> float:
    return 0.0 if score_name in MAXIMIZED_SCORE_NAMES else math.inf


def _format_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{message['role']}]\n{message['content']}" for message in messages
    )
