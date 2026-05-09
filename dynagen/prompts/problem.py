from typing import Any

from dynagen.candidates.candidate import Candidate
from dynagen.prompts.bbob_evolution import build_bbob_evolution_prompt
from dynagen.prompts.bbob_initial import BBOB_INITIAL_ROLES, BBOBInitialRole, build_bbob_initial_prompt
from dynagen.prompts.evolution import build_evolution_prompt
from dynagen.prompts.initial import INITIAL_ROLES, InitialRole, build_initial_prompt


def initial_roles_for_problem(count: int, problem_type: str) -> list[Any]:
    problem_type = problem_type.lower()
    source = BBOB_INITIAL_ROLES if problem_type == "bbob" else INITIAL_ROLES
    roles: list[Any] = []
    for index in range(count):
        role = source[index % len(source)]
        role_type = BBOBInitialRole if problem_type == "bbob" else InitialRole
        roles.append(role_type(index + 1, role.role, role.intended_bias))
    return roles


def build_problem_initial_prompt(role: Any, problem_type: str) -> list[dict[str, str]]:
    if problem_type.lower() == "bbob":
        return build_bbob_initial_prompt(role)
    return build_initial_prompt(role)


def build_problem_evolution_prompt(strategy: str, parents: list[Candidate], problem_type: str) -> list[dict[str, str]]:
    if problem_type.lower() == "bbob":
        return build_bbob_evolution_prompt(strategy, parents)
    return build_evolution_prompt(strategy, parents)
