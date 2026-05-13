from dynagen.candidates.candidate import Candidate
from dynagen.prompts.dvrp_templates import (
    DVRP_INTERNAL_CHECKLIST,
    DVRP_POLICY_CONTRACT,
    DVRP_RESPONSE_FORMAT,
    render_dvrp_candidates,
)

DVRP_STRATEGY_INSTRUCTIONS = {
    "S1": """Explore: produce a dispatch policy whose decision mechanism differs fundamentally from the parent.
First, identify the parent's core decision mechanism in one short phrase — for example scoring, multi-step lookahead, regret comparison, clustering, or fleet-aware coordination.
Then, choose a different mechanism class and design the new policy around it.
Keep validity, deterministic seed handling, and budget-bounded per-call work.""",

    "S2": """Refine: improve the parent through targeted fixes grounded in its measured behavior.
First, identify the parent's weakest measured case from its metrics (mean_gap, mean_makespan, gap_by_instance_size) and its prior thought.
Then, propose one or two focused changes that directly address that weakness.
Preserve what works; avoid unrelated rewrites.""",

    "S3": """Recombine: produce a single coherent policy that captures the common principle behind the parents and introduces new mechanism.
First, identify the common idea or principle the parents share — the underlying mechanism, not their surface features.
Then, describe in one sentence a new policy built on that principle, with components not present in any parent.
Finally, implement it as one decision pass; do not concatenate parents, run them sequentially, vote between them, or branch by instance condition.
Keep the child simpler than the sum.""",
}


def build_dvrp_evolution_prompt(
    strategy: str,
    parents: list[Candidate],
    *,
    generation_reflection: str = "",
) -> list[dict[str, str]]:
    if strategy not in DVRP_STRATEGY_INSTRUCTIONS:
        raise ValueError(f"Unknown strategy: {strategy}")
    """blocks = [
        f"STRATEGY {strategy}: {DVRP_STRATEGY_INSTRUCTIONS[strategy]}",
        "Minimize time until the last truck returns to the depot. This is the only optimization goal.",
    ] """
    blocks = [
    DVRP_POLICY_CONTRACT.strip(),
    DVRP_INTERNAL_CHECKLIST.strip(),
    DVRP_RESPONSE_FORMAT.strip(),
    f"PARENTS:\n{render_dvrp_candidates(parents)}",
    ]
    if generation_reflection:
        blocks.append(f"REFLECTION FROM RECENT PARENT/CHILD COMPARISON:\n{generation_reflection}")

    blocks.extend([
    f"STRATEGY {strategy}: {DVRP_STRATEGY_INSTRUCTIONS[strategy]}",
    "Minimize time until the last truck returns to the depot. This is the only optimization goal.",
    ])
    
    """    
    blocks.extend([
        f"PARENTS:\n{render_dvrp_candidates(parents)}",
        DVRP_POLICY_CONTRACT.strip(),
        DVRP_INTERNAL_CHECKLIST.strip(),
        DVRP_RESPONSE_FORMAT.strip(),
    ])"""
    user = "\n\n".join(blocks)
    return [
        {"role": "system", "content": "You generate compact online DVRP dispatch policies that minimize last-truck return time."},
        {"role": "user", "content": user},
    ]