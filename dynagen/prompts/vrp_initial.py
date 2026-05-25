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


VRP_INITIAL_ROLES = [
    VRPInitialRole(
        slot=1,
        role="Savings-based constructive solver",
        intended_bias=(
            "Build routes using a Clarke-Wright-style savings principle: start with simple "
            "customer-to-depot routes, then merge routes when the merge reduces travel cost "
            "and preserves depot, customer coverage, and vehicle-count constraints."
        ),
    ),
    VRPInitialRole(
        slot=2,
        role="Greedy insertion constraint-aware solver",
        intended_bias=(
            "Construct routes by inserting unrouted customers into the cheapest feasible "
            "position. Prioritize feasibility first, then minimize marginal cost increase. "
            "Use deterministic repair or fallback logic for difficult customers."
        ),
    ),
    VRPInitialRole(
        slot=3,
        role="Regret-insertion solver",
        intended_bias=(
            "Insert customers whose best feasible insertion is much better than their next-best "
            "alternatives. Make hard placement decisions early to avoid leaving expensive or "
            "constraint-sensitive customers until the end."
        ),
    ),
    VRPInitialRole(
        slot=4,
        role="Cluster-first route-second solver",
        intended_bias=(
            "Group geographically or cost-similar customers into feasible route clusters first, "
            "then order each route internally. Bias toward compact routes, balanced vehicle usage, "
            "and reduced cross-route overlap."
        ),
    ),
    VRPInitialRole(
        slot=5,
        role="Construct-and-improve local-search solver",
        intended_bias=(
            "Create a feasible initial solution using a simple robust construction method, then "
            "improve it with local moves such as relocate, swap, 2-opt within routes, and limited "
            "cross-route exchanges while preserving feasibility."
        ),
    ),
]


def build_vrp_initial_prompt(role: VRPInitialRole) -> list[dict[str, str]]:
    system = vrp_system_prompt()

    user = "\n\n".join([
        "# Initial Candidate Identity",
        f"Candidate ID: {role.slot}",
        f"Role: {role.role}",
        f"Intended bias: {role.intended_bias}",

        "# Internal Quality Checklist",
        VRP_INTERNAL_CHECKLIST.strip(),

        "# Solver Contract",
        VRP_SOLVER_CONTRACT.strip(),

        "# Response Format",
        VRP_RESPONSE_FORMAT.strip(),
    ])

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
