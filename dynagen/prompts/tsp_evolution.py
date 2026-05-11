from dynagen.candidates.candidate import Candidate
from dynagen.prompts.tsp_templates import (
    TSP_INTERNAL_CHECKLIST,
    TSP_RESPONSE_FORMAT,
    TSP_SOLVER_CONTRACT,
    render_tsp_candidates,
)

# S1:explore, S2:refine, S3:recombine.
TSP_STRATEGY_INSTRUCTIONS = {
    "S1": """Explore: create a complete solver with materially different search behavior from the parent.
Change a core decision rule or search dynamic, not names or constants. Keep validity, budget use, and early reporting.""",

    "S2": """Refine: use parent metrics and reflection to make one or two targeted fixes.
Preserve what works, address measured weakness, and avoid unrelated rewrites.""",

    "S3": """Recombine: build one coherent solver from complementary parent strengths.
Do not concatenate parents or run them sequentially. Resolve conflicts and keep the child simpler than the sum."""
}


def build_tsp_evolution_prompt(
        strategy: str,
        parents: list[Candidate],
        *,
        generation_reflection: str = "",
) -> list[dict[str, str]]:
    if strategy not in TSP_STRATEGY_INSTRUCTIONS:
        raise ValueError(f"Unknown strategy: {strategy}")
    candidates_context = render_tsp_candidates(parents)
    blocks = [
        f"STRATEGY {strategy}: {TSP_STRATEGY_INSTRUCTIONS[strategy]}",
        "Distance is the search objective for TSP; lower distance is better.",
    ]
    if generation_reflection:
        blocks.append(f"REFLECTION FROM RECENT PARENT/CHILD COMPARISON:\n{generation_reflection}")
    blocks.extend([
        f"PARENTS:\n{candidates_context}",
        TSP_SOLVER_CONTRACT.strip(),
        TSP_INTERNAL_CHECKLIST.strip(),
        TSP_RESPONSE_FORMAT.strip(),
    ])
    user = "\n\n".join(blocks)
    return [
        {"role": "system", "content": "You generate executable, reliable full TSP solvers for evolutionary search."},
        {"role": "user", "content": user},
    ]
