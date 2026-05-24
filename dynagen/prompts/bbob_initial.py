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


BBOB_INITIAL_ROLES = [
    BBOBInitialRole(
        slot=1,
        role="CMA-ES-style covariance adaptation solver",
        intended_bias=(
            "Use a population-based Gaussian search distribution with adaptive step size "
            "and covariance-like directional learning. Bias toward robustness on rotated, "
            "non-separable, ill-conditioned continuous landscapes."
        ),
    ),
    BBOBInitialRole(
        slot=2,
        role="Differential-evolution-style population solver",
        intended_bias=(
            "Maintain a diverse population and generate candidates by vector differences, "
            "crossover, and greedy replacement. Bias toward global exploration on multimodal "
            "and weakly structured landscapes."
        ),
    ),
    BBOBInitialRole(
        slot=3,
        role="Restarted adaptive random search solver",
        intended_bias=(
            "Use randomized sampling around the best incumbent with adaptive radius control "
            "and restarts after stagnation. Bias toward simple, budget-safe anytime performance "
            "across many function types."
        ),
    ),
    BBOBInitialRole(
        slot=4,
        role="Coordinate and pattern-search solver",
        intended_bias=(
            "Probe coordinate directions and simple search patterns with adaptive step sizes. "
            "Bias toward separable, partially separable, low-dimensional, and locally smooth "
            "problems while remaining derivative-free."
        ),
    ),
    BBOBInitialRole(
        slot=5,
        role="Memetic global-local solver",
        intended_bias=(
            "Combine broad global sampling with local refinement around the best candidates. "
            "Bias toward balancing exploration on multimodal functions with exploitation on "
            "unimodal or funnel-like regions."
        ),
    ),
]


def build_bbob_initial_prompt(role: BBOBInitialRole) -> list[dict[str, str]]:
    system = bbob_system_prompt()

    user = "\n\n".join([
        "# Initial Candidate Identity",
        f"Candidate ID: {role.slot}",
        f"Role: {role.role}",
        f"Intended bias: {role.intended_bias}",

        "# DynaGen Scoring",
        "DynaGen score: mean AOCC; higher is better. The Optimizer itself still minimizes objective values.",

        "# Internal Quality Checklist",
        BBOB_INTERNAL_CHECKLIST.strip(),

        "# Solver Contract",
        BBOB_SOLVER_CONTRACT.strip(),

        "# Response Format",
        BBOB_RESPONSE_FORMAT.strip(),

    ])

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
