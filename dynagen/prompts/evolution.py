from __future__ import annotations

from dynagen.candidates.candidate import Candidate
from dynagen.prompts.templates import RESPONSE_FORMAT, SOLVER_CONTRACT, render_candidates

STRATEGY_INSTRUCTIONS = {
    "E1": """Divergent Exploration: generate a full TSP solver that differs substantially from the selected parents. Do not merely tune parameters; explore a different algorithmic family or hybrid while preserving the contract.""",
    "E2": """Backbone Exploration: infer the common successful ideas from strong parents, keep the useful backbone, and add a new mechanism or search behavior.""",
    "E3": """Two-Parent Crossover: combine useful traits from the selected parents into one coherent full solver while avoiding a simple parameter-only variant.""",
    "M1": """Performance Mutation: improve the selected parent using concrete evaluation feedback. Fix reliability problems if present and keep the implementation compact and robust.""",
    "M2": """Parameter And Operator Mutation: preserve the parent design while changing schedules, acceptance criteria, neighborhood choice, restart policy, mutation strength, or local-search depth.""",
}


def build_evolution_prompt(strategy: str, parents: list[Candidate]) -> list[dict[str, str]]:
    if strategy not in STRATEGY_INSTRUCTIONS:
        raise ValueError(f"Unknown strategy: {strategy}")
    candidates_context = render_candidates(parents)
    user = f"""
    STRATEGY: {strategy}
    {STRATEGY_INSTRUCTIONS[strategy]}
    
    SELECTED PARENT(S) CONTEXT:
    {candidates_context}
    
    {SOLVER_CONTRACT}
    
    {RESPONSE_FORMAT}"""
    return [
        {"role": "system", "content": "You generate executable, reliable full TSP solvers for evolutionary search."},
        {"role": "user", "content": user},
    ]
