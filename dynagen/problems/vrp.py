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
            timeout_seconds=config.evaluation.timeout_seconds,
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


def create_vrp_initial_roles(count: int) -> list[VRPInitialRole]:
    roles: list[VRPInitialRole] = []
    for index in range(count):
        role = VRP_INITIAL_ROLES[index % len(VRP_INITIAL_ROLES)]
        roles.append(VRPInitialRole(index + 1, role.role, role.intended_bias))
    return roles
