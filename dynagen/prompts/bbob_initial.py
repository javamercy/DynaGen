from dataclasses import dataclass

from dynagen.prompts.bbob_templates import (
    BBOB_INTERNAL_CHECKLIST,
    BBOB_RESPONSE_FORMAT,
    BBOB_SOLVER_CONTRACT,
    bbob_system_prompt,
)


@dataclass(frozen=True)
class BBOBInitialRole:
    slot: int
    role: str
    intended_bias: str


BBOB_INITIAL_ROLES = (
    BBOBInitialRole(
        1,
        "a scale-adaptive evolution strategy designer",
        "Design an optimizer inspired by strong evolution-strategy principles: adaptive sampling radius, elite selection, success-based updates, restart behavior, and dimension-aware scaling.",
    ),
    BBOBInitialRole(
        2,
        "a population recombination optimizer engineer",
        "Design a compact population-based optimizer using mutation, recombination, replacement, and diversity control to make reliable progress across separable, conditioned, and multimodal functions.",
    ),
    BBOBInitialRole(
        3,
        "a restart-based global-local search strategist",
        "Design an optimizer that combines broad exploration, local improvement, stagnation detection, and restarts or radius resets to escape poor basins while preserving the best incumbent.",
    ),
    BBOBInitialRole(
        4,
        "a coordinate and pattern search specialist",
        "Design an optimizer that exploits coordinate-wise, directional, or pattern-based local moves with adaptive step sizes and bounded evaluation discipline.",
    ),
    BBOBInitialRole(
        5,
        "a hybrid adaptive optimizer architect",
        "Design a compact hybrid optimizer that combines complementary mechanisms such as population sampling, local refinement, success memory, adaptive step control, and restart logic.",
    ),
)


def build_bbob_initial_prompt(role: BBOBInitialRole) -> list[dict[str, str]]:
    user = f"""Generate initial population slot {role.slot}.

Assigned optimizer perspective:
{role.role}

Behavioral bias:
{role.intended_bias}

{BBOB_SOLVER_CONTRACT}

{BBOB_INTERNAL_CHECKLIST}

{BBOB_RESPONSE_FORMAT}"""
    return [
        {"role": "system", "content": bbob_system_prompt(role.role)},
        {"role": "user", "content": user},
    ]
