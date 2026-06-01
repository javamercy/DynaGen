import logging
from pathlib import Path
from typing import Any

from dynagen.candidates.candidate import Candidate
from dynagen.config import RunConfig
from dynagen.domain import load_tsplib_file
from dynagen.domain.tsp_instance import TSPInstance
from dynagen.domain.tsp_synthetic import (
    generate_tsp_construct_instances,
    parse_tsp_construct_spec,
)
from dynagen.evaluation.tsp_history import build_tsp_history_profile
from dynagen.evaluation.tsp_gradient import build_tsp_llm_verbal_gradient_prompt
from dynagen.evaluation.tsp_evaluator import TSPCandidateEvaluator
from dynagen.prompts.tsp_evolution import build_tsp_evolution_prompt
from dynagen.prompts.tsp_initial import (
    TSP_INITIAL_ROLES,
    TSPInitialRole,
    build_tsp_initial_prompt,
)


logger = logging.getLogger(__name__)


class TSPProblem:
    type = "tsp"

    def build_evaluator(
        self, config: RunConfig, *, pool_name: str
    ) -> TSPCandidateEvaluator:
        path = (
            config.data.search_instances
            if pool_name == "search_instances"
            else config.data.test_instances
        )
        logger.info(
            "[%s] initializing %s pool from %s", self.type.upper(), pool_name, path
        )
        instances = load_tsp_instances(path)
        logger.info(
            "[%s] loaded %d instances for %s",
            self.type.upper(),
            len(instances),
            pool_name,
        )
        return TSPCandidateEvaluator(
            instances,
            timeout_seconds=config.evaluation.timeout_seconds,
            timeout_penalty=config.evaluation.timeout_penalty,
            pool_name=pool_name,
        )

    def initial_roles(self, count: int) -> list[TSPInitialRole]:
        return create_tsp_initial_roles(count)

    def build_initial_prompt(self, role: Any) -> list[dict[str, str]]:
        return build_tsp_initial_prompt(role)

    def build_evolution_prompt(
        self,
        strategy: str,
        parents: list[Candidate],
        *,
        feedback_context: str = "",
    ) -> list[dict[str, str]]:
        return build_tsp_evolution_prompt(
            strategy,
            parents,
            feedback_context=feedback_context,
        )

    def build_llm_verbal_gradient_prompt(
        self,
        candidate: Candidate,
        *,
        parents: list[Candidate],
        generation: int,
    ) -> list[dict[str, str]]:
        return build_tsp_llm_verbal_gradient_prompt(
            candidate,
            parents=parents,
            generation=generation,
        )

    def build_history_profile(self, candidate: Candidate) -> dict[str, Any]:
        return build_tsp_history_profile(candidate)

    def per_instance_scores(self, candidate: Candidate) -> dict[str, float]:
        metrics = candidate.metrics if isinstance(candidate.metrics, dict) else {}
        scores: dict[str, float] = {}
        by_size = metrics.get("score_by_instance_size")
        if isinstance(by_size, dict):
            for k, v in by_size.items():
                scores[f"size:{k}"] = _invert_tsp_score(v)
        by_source = metrics.get("score_by_instance_source")
        if isinstance(by_source, dict):
            for k, v in by_source.items():
                scores[f"source:{k}"] = _invert_tsp_score(v)
        return scores


def _invert_tsp_score(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not number == number or number in {float("inf"), float("-inf")}:
        return 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, number) / 25.0)))


def create_tsp_initial_roles(count: int) -> list[TSPInitialRole]:
    roles: list[TSPInitialRole] = []
    for index in range(count):
        role = TSP_INITIAL_ROLES[index % len(TSP_INITIAL_ROLES)]
        roles.append(TSPInitialRole(index + 1, role.role, role.intended_bias))
    return roles


def load_tsp_instances(path: str | Path | None) -> list[TSPInstance]:
    if not path:
        raise ValueError(
            "TSP data.search_instances and data.test_instances must be specified"
        )

    synthetic_spec = parse_tsp_construct_spec(str(path))
    if synthetic_spec is not None:
        n_instance, n_cities, seed = synthetic_spec
        return generate_tsp_construct_instances(
            n_instance=n_instance,
            n_cities=n_cities,
            seed=seed,
        )

    path = Path(path)
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.suffix.lower() == ".tsp")
        if not files:
            raise ValueError(f"No .tsp files found in {path}")
        return [load_tsplib_file(file) for file in files]
    return [load_tsplib_file(path)]
