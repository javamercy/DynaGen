from typing import Any

from dynagen.candidates.candidate import Candidate
from dynagen.config import RunConfig
from dynagen.domain.bbob import BBOBInstance, create_bbob_instances
from dynagen.evaluation.bbob_evaluator import BBOBCandidateEvaluator
from dynagen.prompts.bbob_evolution import build_bbob_evolution_prompt
from dynagen.prompts.bbob_initial import BBOB_INITIAL_ROLES, BBOBInitialRole, build_bbob_initial_prompt


class BBOBProblem:
    type = "bbob"

    def build_evaluator(self, config: RunConfig, *, pool_name: str) -> BBOBCandidateEvaluator:
        return BBOBCandidateEvaluator(
            load_bbob_instances(config, pool_name=pool_name),
            seeds=config.evaluation.seeds,
            budget=config.evaluation.budget,
            timeout_seconds=config.evaluation.timeout_seconds,
            timeout_penalty=config.evaluation.timeout_penalty,
            pool_name=pool_name,
            aocc_lower_bound=config.problem.aocc_lower_bound,
            aocc_upper_bound=config.problem.aocc_upper_bound,
        )

    def initial_roles(self, count: int) -> list[BBOBInitialRole]:
        return create_bbob_initial_roles(count)

    def build_initial_prompt(self, role: Any) -> list[dict[str, str]]:
        return build_bbob_initial_prompt(role)

    def build_evolution_prompt(self, strategy: str, parents: list[Candidate]) -> list[dict[str, str]]:
        return build_bbob_evolution_prompt(strategy, parents)


def create_bbob_initial_roles(count: int) -> list[BBOBInitialRole]:
    roles: list[BBOBInitialRole] = []
    for index in range(count):
        role = BBOB_INITIAL_ROLES[index % len(BBOB_INITIAL_ROLES)]
        roles.append(BBOBInitialRole(index + 1, role.role, role.intended_bias))
    return roles


def load_bbob_instances(config: RunConfig, *, pool_name: str) -> list[BBOBInstance]:
    if pool_name == "search_instances":
        instance_ids = config.problem.search_instances
        dimensions = [config.problem.dimension]
    else:
        instance_ids = config.problem.test_instances
        dimensions = config.problem.test_dimensions
    return create_bbob_instances(
        function_ids=config.problem.function_ids,
        instance_ids=instance_ids,
        dimensions=dimensions,
        bounds=config.problem.bounds,
    )
