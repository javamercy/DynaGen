import math
import random
import traceback

from dynagen.candidates import CandidateStatus, ParsedCandidateResponse
from dynagen.candidates.candidate import Candidate
from dynagen.config import RunConfig
from dynagen.evaluation.evaluator import CandidateEvaluator
from dynagen.evaluation.metrics import aggregate_records
from dynagen.evolution.population import Population
from dynagen.evolution.selection import select_parents, select_survivors
from dynagen.evolution.strategies import parent_count, Strategy
from dynagen.llm.base import LLMProvider
from dynagen.persistence.run_store import RunStore
from dynagen.prompts.evolution import build_evolution_prompt
from dynagen.prompts.initial import INITIAL_ROLES, InitialRole, build_initial_prompt
from dynagen.reporting.summary import build_final_report, generation_summary


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
        self.rng = random.Random(config.seed)

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
        self.store.write_final_report(
            build_final_report(
                population.candidates,
                search_best=search_best,
                test_result=test_result,
            )
        )
        return population

    def _initial_population(self) -> Population:
        candidates: list[Candidate] = []
        for role in _initial_roles(self.config.evolution.population_size):
            messages = build_initial_prompt(role)
            prompt = _format_messages(messages)
            candidate_id = self.store.next_candidate_id()
            candidate: Candidate | None = None
            try:
                response = self.provider.complete(
                    messages,
                    temperature=self.config.llm.temperature,
                )
                candidate = _build_candidate_from_response(
                    response,
                    candidate_id=candidate_id,
                    generation=0,
                    strategy=f"initial:{role.slot}",
                    prompt=prompt,
                )
                self.search_evaluator.evaluate_candidate(candidate)
            except Exception:
                error_details = _exception_details()
                if candidate is None:
                    candidate = _failed_candidate(
                        candidate_id=candidate_id,
                        generation=0,
                        strategy=f"initial:{role.slot}",
                        prompt=prompt,
                        error_details=error_details,
                    )
                else:
                    _mark_candidate_error(candidate, error_details)
            self.store.save_candidate(candidate)
            candidates.append(candidate)
        return Population.from_candidates(0, candidates, size=self.config.evolution.population_size)

    def _generate_offspring(self, generation: int, population: Population) -> list[Candidate]:
        offspring: list[Candidate] = []
        for strategy in self.config.evolution.strategies:
            for _ in range(self.config.evolution.offspring_per_strategy):
                candidate_id = self.store.next_candidate_id()
                parents: list[Candidate] = []
                prompt = ""
                candidate: Candidate | None = None
                try:
                    parents = self._select_strategy_parents(strategy, population.candidates)
                    messages = build_evolution_prompt(strategy, parents)
                    prompt = _format_messages(messages)
                    response = self.provider.complete(
                        messages,
                        temperature=self.config.llm.temperature,
                    )
                    candidate = _build_candidate_from_response(
                        response,
                        candidate_id=candidate_id,
                        generation=generation,
                        strategy=strategy,
                        parents=[parent.id for parent in parents],
                        prompt=prompt,
                    )
                    self.search_evaluator.evaluate_candidate(candidate)
                except Exception:
                    error_details = _exception_details()
                    if candidate is None:
                        candidate = _failed_candidate(
                            candidate_id=candidate_id,
                            generation=generation,
                            strategy=strategy,
                            parents=[parent.id for parent in parents],
                            prompt=prompt,
                            error_details=error_details,
                        )
                    else:
                        _mark_candidate_error(candidate, error_details)
                self.store.save_candidate(candidate)
                offspring.append(candidate)
        return offspring

    def _select_strategy_parents(self, strategy: Strategy, candidates: list[Candidate]) -> list[Candidate]:
        return select_parents(candidates, parent_count(strategy), self.rng)


def _initial_roles(count: int) -> list[InitialRole]:
    roles: list[InitialRole] = []
    for index in range(count):
        role = INITIAL_ROLES[index % len(INITIAL_ROLES)]
        roles.append(InitialRole(index + 1, role.role, role.intended_bias))
    return roles


def _build_candidate_from_response(
        response: ParsedCandidateResponse,
        *,
        candidate_id: str,
        generation: int,
        strategy: str,
        parents: list[str] | None = None,
        prompt: str,
) -> Candidate:
    return Candidate(
        id=candidate_id,
        generation=generation,
        strategy=strategy,
        name=response.name,
        thought=response.thought,
        code=response.code,
        parents=list(parents or []),
        fitness=None,
        metrics=aggregate_records([]),
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
) -> Candidate:
    return Candidate(
        id=candidate_id,
        generation=generation,
        strategy=strategy,
        name="",
        thought="",
        code="",
        parents=list(parents or []),
        fitness=math.inf,
        metrics=aggregate_records([]),
        status=CandidateStatus.ERROR,
        prompt=prompt,
        error_details=error_details,
    )


def _exception_details() -> str:
    return traceback.format_exc(limit=20).strip()


def _mark_candidate_error(candidate: Candidate, error_details: str) -> None:
    candidate.status = CandidateStatus.ERROR
    candidate.fitness = math.inf
    if not candidate.metrics:
        candidate.metrics = aggregate_records([])
    candidate.error_details = error_details


def _format_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{message['role']}]\n{message['content']}"
        for message in messages
    )
