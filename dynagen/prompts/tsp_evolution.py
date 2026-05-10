from dynagen.candidates.candidate import Candidate
from dynagen.prompts.tsp_templates import (
    TSP_INTERNAL_CHECKLIST,
    TSP_RESPONSE_FORMAT,
    TSP_SOLVER_CONTRACT,
    render_tsp_candidates,
)

# S1:explore, S2:improve, S3:combine, S4:stabilize.
TSP_STRATEGY_INSTRUCTIONS = {
    "S1": """Behavioral Innovation:
Generate a complete TSP solver with a substantially different search behavior from the selected parent(s).

Do not merely tune parameters, rename variables, or rearrange the same algorithm.
Do not default to a generic nearest-neighbor plus plain local-search solver unless it contains a genuinely distinct mechanism.

The new solver should differ from the parent(s) in at least two of these dimensions:
- how it constructs or initializes tours
- how it chooses candidate modifications
- how it accepts, rejects, or ranks changes
- how it escapes poor local minima
- how it allocates budget over time
- how it uses seed-controlled randomness
- how it remembers useful or harmful choices during the run
- how it exploits structure in the distance matrix

You may invent, hybridize, simplify, or adapt mechanisms.
Prioritize validity and robustness, but make the search behavior meaningfully novel.""",

    "S2": """Evidence-Guided Performance Upgrade:
Improve the selected parent into a stronger complete solver while preserving its useful backbone.

Use the parent context as evidence. Keep what appears reliable or effective, and change what appears weak, wasteful, brittle, or under-explored.
The offspring should improve at least one of:
- tour quality
- quality per unit budget
- robustness to timeout
- ability to escape stagnation
- effective use of seed-controlled randomness
- efficient use of distance-matrix structure

Do not make a parameter-only mutation unless the parent is already strong and the change directly improves budget allocation or reliability.
Prefer one or two coherent mechanism changes over many shallow tweaks.
The result should still be compact, valid, and easy for the evaluator to execute.""",

    "S3": """Complementary Mechanism Recombination:
Combine useful traits from selected parents into one coherent complete solver.

Do not concatenate multiple solvers or run parent algorithms sequentially without integration.
Choose one clear backbone, then integrate one or two complementary mechanisms from the other parent(s).
Useful complementary mechanisms may involve:
- construction behavior
- improvement behavior
- diversification or transition behavior
- memory or bias accumulated during the run
- budget allocation
- candidate selection or pruning
- robustness and reporting behavior

The offspring should be simpler than the sum of its parents.
Resolve conflicts between parent designs rather than including both versions.
Avoid code bloat, duplicated logic, and excessive nested loops.""",

    "S4": """Robust Anytime Simplification:
Generate a complete solver focused on reliability, valid incumbents, and improvement under strict budget.

Use this strategy when parents are brittle, timeout-prone, overly complex, invalid, or inconsistent.
The offspring must:
- create a valid incumbent early
- report the initial incumbent and every improvement
- return a valid tour on every path
- use budget to bound expensive work
- avoid unguarded high-complexity loops
- preserve only mechanisms that clearly contribute to search progress

This is not merely a repair strategy. After making the solver robust, improve its quality-per-budget using compact search logic.
Prefer a smaller reliable solver over a large fragile one."""
}


def build_tsp_evolution_prompt(strategy: str, parents: list[Candidate]) -> list[dict[str, str]]:
    if strategy not in TSP_STRATEGY_INSTRUCTIONS:
        raise ValueError(f"Unknown strategy: {strategy}")
    candidates_context = render_tsp_candidates(parents)
    user = f"""
    STRATEGY: {strategy}
    {TSP_STRATEGY_INSTRUCTIONS[strategy]}
    
    SELECTED PARENT(S) CONTEXT:
    {candidates_context}
    
    {TSP_SOLVER_CONTRACT}
    
    {TSP_INTERNAL_CHECKLIST}
    
    {TSP_RESPONSE_FORMAT}"""
    return [
        {"role": "system", "content": "You generate executable, reliable full TSP solvers for evolutionary search."},
        {"role": "user", "content": user},
    ]
