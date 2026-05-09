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
        "a landscape-probing optimizer designer",
        "Design an optimizer that samples enough early points to infer rough scale, conditioning, separability, and multimodality signals before committing budget to improvement.",
    ),
    BBOBInitialRole(
        2,
        "an adaptive population search engineer",
        "Design a compact population-based optimizer with budget-aware mutation, recombination, and replacement. Emphasize reliable progress across separable, conditioned, and multimodal functions.",
    ),
    BBOBInitialRole(
        3,
        "a restart and diversification strategist",
        "Design an optimizer that balances local exploitation with restarts, radius changes, or distribution resets when progress stagnates.",
    ),
    BBOBInitialRole(
        4,
        "a covariance-free local adaptation researcher",
        "Design an optimizer that adapts coordinate scales, step sizes, and directional biases without relying on external libraries or heavy matrix operations.",
    ),
    BBOBInitialRole(
        5,
        "a memory-guided continuous search architect",
        "Design an optimizer that remembers successful moves, directions, coordinates, or population statistics and uses them to bias later proposals under the evaluation budget.",
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
