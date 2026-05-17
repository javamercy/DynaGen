from dynagen.candidates.candidate import Candidate
from dynagen.prompts.parent_awareness import render_parent_awareness
from dynagen.prompts.vrp_templates import (
    VRP_INTERNAL_CHECKLIST,
    VRP_RESPONSE_FORMAT,
    VRP_SOLVER_CONTRACT,
    render_vrp_candidates,
)

VRP_STRATEGY_INSTRUCTIONS = {
    "S1": """Explore: create a complete VRP solver with materially different route-construction or improvement behavior from the parent.
Change a core mechanism such as clustering, savings, split assignment, local-search neighborhoods, restart dynamics, or population structure.
Keep feasibility, deterministic seed handling, report_best_vrp, and budget-bounded search.""",

    "S2": """Refine: use parent metrics and LLM reflections to make one or two targeted fixes.
Focus on the measured weakness: poor large-instance gap, poor truck-count behavior, high max-route distance, timeouts, invalid routes, or weak route balance.
Preserve what works; avoid unrelated rewrites.""",

    "S3": """Recombine: build one coherent VRP solver from complementary parent strengths.
Choose a single backbone, then integrate one or two compatible mechanisms from other parents.
Do not concatenate solvers, vote between parents, or run full parents sequentially. The child must be simpler than the sum.""",
}


def build_vrp_evolution_prompt(
        strategy: str,
        parents: list[Candidate],
        *,
        feedback_context: str = "",
) -> list[dict[str, str]]:
    if strategy not in VRP_STRATEGY_INSTRUCTIONS:
        raise ValueError(f"Unknown strategy: {strategy}")
    parent_awareness = render_parent_awareness(
        parents,
        strategy=strategy,
        problem="vrp",
        score_label="distance",
    )
    blocks = [
        f"STRATEGY {strategy}: {VRP_STRATEGY_INSTRUCTIONS[strategy]}",
        "VRP objective: lower maximum route distance across trucks is better. Feasibility is mandatory.",
    ]
    if feedback_context:
        blocks.append(feedback_context)
    if parent_awareness:
        blocks.append(parent_awareness)
    blocks.extend([
        f"PARENTS:\n{render_vrp_candidates(parents)}",
        VRP_SOLVER_CONTRACT.strip(),
        VRP_INTERNAL_CHECKLIST.strip(),
        VRP_RESPONSE_FORMAT.strip(),
    ])
    return [
        {"role": "system", "content": "You generate executable, reliable VRP metaheuristics for evolutionary search."},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]
