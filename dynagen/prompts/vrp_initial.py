from dataclasses import dataclass

from dynagen.prompts.vrp_templates import (
    VRP_INTERNAL_CHECKLIST,
    VRP_RESPONSE_FORMAT,
    VRP_SOLVER_CONTRACT,
    vrp_system_prompt,
)


@dataclass(frozen=True)
class VRPInitialRole:
    slot: int
    role: str
    intended_bias: str


VRP_INITIAL_ROLES = (
    VRPInitialRole(
        1,
        "a fleet-balance constructive metaheuristic designer",
        "Build routes by assigning the next customer to the truck where it least increases the current maximum route distance.",
    ),
    VRPInitialRole(
        2,
        "a cluster-first route-second strategist",
        "Partition customers into truck-sized regions, route each cluster, then rebalance border customers.",
    ),
    VRPInitialRole(
        3,
        "a savings-and-repair optimizer",
        "Use merge savings to form routes, then repair overload in max-route distance with relocations and swaps.",
    ),
    VRPInitialRole(
        4,
        "a local-search intensive route improver",
        "Start from a feasible split and spend budget on 2-opt, relocate, exchange, and max-route balancing moves.",
    ),
    VRPInitialRole(
        5,
        "a restart-based anytime search designer",
        "Generate diverse feasible route sets under seed control and keep the best minimax incumbent via report_best_vrp.",
    ),
)


def build_vrp_initial_prompt(role: VRPInitialRole) -> list[dict[str, str]]:
    user = "\n\n".join([
        f"Initial slot {role.slot}",
        f"Perspective: {role.role}\nBias: {role.intended_bias}",
        VRP_SOLVER_CONTRACT.strip(),
        "Optimization goal: minimize the longest truck route, not only total distance.",
        VRP_INTERNAL_CHECKLIST.strip(),
        VRP_RESPONSE_FORMAT.strip(),
    ])
    return [
        {"role": "system", "content": vrp_system_prompt(role.role)},
        {"role": "user", "content": user},
    ]
