import logging
from typing import Any

from dynagen.candidates.candidate import Candidate
from dynagen.config import RunConfig
from dynagen.domain.vrp import load_vrp_instances
from dynagen.evaluation.vrp_history import build_vrp_history_profile
from dynagen.evaluation.vrp_evaluator import VRPCandidateEvaluator
from dynagen.evaluation.vrp_gradient import build_vrp_llm_verbal_gradient_prompt
from dynagen.prompts.vrp_evolution import build_vrp_evolution_prompt
from dynagen.prompts.vrp_initial import VRP_INITIAL_ROLES, VRPInitialRole, build_vrp_initial_prompt

logger = logging.getLogger(__name__)


class VRPProblem:
    type = "vrp"

    def build_evaluator(self, config: RunConfig, *, pool_name: str) -> VRPCandidateEvaluator:
        path = config.data.search_instances if pool_name == "search_instances" else config.data.test_instances
        logger.info("[%s] initializing %s pool from %s", self.type.upper(), pool_name, path)
        instances = load_vrp_instances(
            path,
            pool_name=pool_name,
            search_limit=config.problem.vrp_search_limit,
            test_sizes=config.problem.vrp_test_sizes,
            test_limit_per_size=config.problem.vrp_test_limit_per_size,
        )
        logger.info("[%s] loaded %d instances for %s", self.type.upper(), len(instances), pool_name)
        return VRPCandidateEvaluator(
            instances,
            timeout_seconds=None if pool_name == "test_instances" else config.evaluation.timeout_seconds,
            timeout_penalty=config.evaluation.timeout_penalty,
            pool_name=pool_name,
        )

    def initial_roles(self, count: int) -> list[VRPInitialRole]:
        return create_vrp_initial_roles(count)

    def build_initial_prompt(self, role: Any) -> list[dict[str, str]]:
        return build_vrp_initial_prompt(role)

    def build_evolution_prompt(
            self,
            strategy: str,
            parents: list[Candidate],
            *,
            feedback_context: str = "",
    ) -> list[dict[str, str]]:
        return build_vrp_evolution_prompt(
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
        return build_vrp_llm_verbal_gradient_prompt(
            candidate,
            parents=parents,
            generation=generation,
        )

    def build_history_profile(self, candidate: Candidate) -> dict[str, Any]:
        return build_vrp_history_profile(candidate)

    def per_instance_scores(self, candidate: Candidate) -> dict[str, float]:
        metrics = candidate.metrics if isinstance(candidate.metrics, dict) else {}
        scores: dict[str, float] = {}
        by_size = metrics.get("score_by_instance_size")
        if isinstance(by_size, dict):
            for k, v in by_size.items():
                scores[f"size:{k}"] = _invert_vrp_score(v)
        by_trucks = metrics.get("score_by_truck_count")
        if isinstance(by_trucks, dict):
            for k, v in by_trucks.items():
                scores[f"trucks:{k}"] = _invert_vrp_score(v)
        by_source = metrics.get("score_by_instance_source")
        if isinstance(by_source, dict):
            for k, v in by_source.items():
                scores[f"source:{k}"] = _invert_vrp_score(v)
        return scores


def _invert_vrp_score(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not number == number or number in {float("inf"), float("-inf")}:
        return 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, number) / 25.0)))


def create_vrp_initial_roles(count: int) -> list[VRPInitialRole]:
    roles: list[VRPInitialRole] = []
    for index in range(count):
        role = VRP_INITIAL_ROLES[index % len(VRP_INITIAL_ROLES)]
        roles.append(VRPInitialRole(index + 1, role.role, role.intended_bias))
    return roles
