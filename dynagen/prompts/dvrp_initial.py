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
        "a spatiotemporal structure exploiter",
        "Infer spatial clusters and temporal patterns from the snapshot to bias the current decision toward structurally good choices.",
    ),
    DVRPInitialRole(
        2,
        "an anticipatory insertion planner",
        "Estimate likely future customer reveals from current geometry and hedge the current choice against them.",
    ),
    DVRPInitialRole(
        3,
        "a multi-step lookahead planner",
        "Use budget to simulate several steps ahead per candidate choice and pick the one with the best estimated makespan contribution.",
    ),
    DVRPInitialRole(
        4,
        "a fleet-wide coordination strategist",
        "Decide for the active truck by reasoning about what other trucks would plausibly do, not in isolation.",
    ),
    DVRPInitialRole(
        5,
        "a regret-based adaptive decider",
        "Score alternatives by estimated regret of committing now versus deferring, and use budget to refine the estimate.",
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
