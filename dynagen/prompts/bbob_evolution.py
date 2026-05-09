from dynagen.candidates.candidate import Candidate
from dynagen.prompts.bbob_templates import (
    BBOB_INTERNAL_CHECKLIST,
    BBOB_RESPONSE_FORMAT,
    BBOB_SOLVER_CONTRACT,
    render_bbob_candidates,
)


BBOB_STRATEGY_INSTRUCTIONS = {
    "S1": """Behavioral Innovation:
Generate a complete black-box optimizer with substantially different search behavior from the selected parent(s).

Do not merely tune constants, rename variables, or rearrange the same algorithm.
The new optimizer should differ from the parent(s) in at least two of these dimensions:
- how it initializes or samples early points
- how it proposes candidate points
- how it adapts step sizes, population statistics, or coordinate scales
- how it handles stagnation and restarts
- how it allocates the function-evaluation budget over time
- how it balances exploration and exploitation
- how it uses seed-controlled randomness

Prioritize budget correctness and robustness, but make the search behavior meaningfully novel.""",

    "S2": """Evidence-Guided Performance Upgrade:
Improve the selected parent into a stronger complete optimizer while preserving its useful backbone.

Use the parent context as evidence. Keep what appears reliable or effective, and change what appears weak, wasteful, brittle, or under-explored.
The offspring should improve at least one of:
- anytime progress measured by best-so-far values
- final objective value under the budget
- robustness to conditioned or multimodal landscapes
- ability to escape stagnation
- effective use of seed-controlled randomness
- safe handling of budget and bounds

Prefer one or two coherent mechanism changes over many shallow tweaks.""",

    "S3": """Complementary Mechanism Recombination:
Combine useful traits from selected parents into one coherent complete optimizer.

Do not concatenate multiple optimizers without integration.
Choose one clear backbone, then integrate one or two complementary mechanisms from the other parent(s).
Useful complementary mechanisms may involve:
- population initialization
- mutation or recombination behavior
- local refinement behavior
- restart or diversification behavior
- memory or adaptive parameter control
- budget allocation

The offspring should be simpler than the sum of its parents.""",

    "S4": """Robust Anytime Simplification:
Generate a complete optimizer focused on reliability, feasible incumbents, and steady improvement under strict budget.

Use this strategy when parents are brittle, timeout-prone, overly complex, invalid, or inconsistent.
The offspring must:
- evaluate a feasible incumbent early
- report the initial incumbent and every improvement
- never exceed the function-evaluation budget
- keep all points inside bounds
- avoid unguarded expensive loops
- preserve only mechanisms that clearly contribute to search progress

This is not merely a repair strategy. After making the optimizer robust, improve quality-per-budget with compact search logic.""",
}


def build_bbob_evolution_prompt(strategy: str, parents: list[Candidate]) -> list[dict[str, str]]:
    if strategy not in BBOB_STRATEGY_INSTRUCTIONS:
        raise ValueError(f"Unknown strategy: {strategy}")
    candidates_context = render_bbob_candidates(parents)
    user = f"""
    STRATEGY: {strategy}
    {BBOB_STRATEGY_INSTRUCTIONS[strategy]}

    SELECTED PARENT(S) CONTEXT:
    {candidates_context}

    {BBOB_SOLVER_CONTRACT}

    {BBOB_INTERNAL_CHECKLIST}

    {BBOB_RESPONSE_FORMAT}"""
    return [
        {"role": "system", "content": "You generate executable, reliable continuous black-box optimizers for evolutionary search."},
        {"role": "user", "content": user},
    ]
