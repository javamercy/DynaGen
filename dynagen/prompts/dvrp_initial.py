from dataclasses import dataclass

from dynagen.prompts.dvrp_templates import (
    DVRP_INTERNAL_CHECKLIST,
    DVRP_POLICY_CONTRACT,
    DVRP_RESPONSE_FORMAT,
    dvrp_system_prompt,
)


@dataclass(frozen=True)
class DVRPInitialRole:
    slot: int
    role: str
    intended_bias: str


DVRP_INITIAL_ROLES = (
    DVRPInitialRole(
        1,
        "a greedy distance minimizer",
        "Choose the customer that most reduces immediate and near-term travel.",
    ),
    DVRPInitialRole(
        2,
        "a fleet balancer",
        "Prevent one truck from becoming much worse than the others when scores are close.",
    ),
    DVRPInitialRole(
        3,
        "a wait minimizer",
        "Wait only if serving now is likely to be clearly worse than waiting briefly.",
    ),
    DVRPInitialRole(
        4,
        "a cluster follower",
        "Prefer nearby customers that keep routes compact.",
    ),
    DVRPInitialRole(
        5,
        "a depot-aware finisher",
        "Keep routes short and avoid choices that create expensive returns.",
    ),
)


def build_dvrp_initial_prompt(role: DVRPInitialRole) -> list[dict[str, str]]:
    user = "\n\n".join([
        f"Initial slot {role.slot}",
        f"Perspective: {role.role}\nBias: {role.intended_bias}",
        DVRP_POLICY_CONTRACT.strip(),
        DVRP_INTERNAL_CHECKLIST.strip(),
        DVRP_RESPONSE_FORMAT.strip(),
    ])
    return [
        {"role": "system", "content": dvrp_system_prompt(role.role)},
        {"role": "user", "content": user},
    ]
