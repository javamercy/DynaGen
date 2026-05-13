from dynagen.candidates.candidate import Candidate
from dynagen.prompts.dvrp_templates import (
    DVRP_INTERNAL_CHECKLIST,
    DVRP_POLICY_CONTRACT,
    DVRP_RESPONSE_FORMAT,
    render_dvrp_candidates,
)


DVRP_STRATEGY_INSTRUCTIONS = {
    "S1": """Explore: create a materially different dispatch rule from the parent.
Use a new customer ranking idea such as spatial zones, customer isolation, depot direction, or truck-competition penalties.
Do not just rename variables or slightly change constants.""",

    "S2": """Mutate and tune: preserve the parent's main structure but improve its scoring formula.
Adjust weights, thresholds, tie-breaks, and wait behavior to reduce the last-truck return time.
Make one or two focused changes rather than rewriting the whole policy.""",

    "S3": """Recombine and simplify: identify the strongest useful idea from each parent and merge them into one compact rule.
Remove redundant conditions, avoid overfit constants, and resolve conflicts into a single cheap score.
Do not concatenate policies or run multiple policies sequentially.""",
}


def build_dvrp_evolution_prompt(
        strategy: str,
        parents: list[Candidate],
        *,
        feedback_context: str = "",
) -> list[dict[str, str]]:
    if strategy not in DVRP_STRATEGY_INSTRUCTIONS:
        raise ValueError(f"Unknown strategy: {strategy}")
    blocks = [
        f"STRATEGY {strategy}: {DVRP_STRATEGY_INSTRUCTIONS[strategy]}",
        "Minimize time until the last truck returns to the depot. This is the only optimization goal.",
        "When customers are available, prefer assigning one instead of waiting unless waiting is clearly better.",
    ]
    if feedback_context:
        blocks.append(feedback_context)
    blocks.extend([
        f"PARENTS:\n{render_dvrp_candidates(parents)}",
        DVRP_POLICY_CONTRACT.strip(),
        DVRP_INTERNAL_CHECKLIST.strip(),
        DVRP_RESPONSE_FORMAT.strip(),
    ])
    user = "\n\n".join(blocks)
    return [
        {"role": "system", "content": "You generate compact online DVRP dispatch policies that minimize last-truck return time."},
        {"role": "user", "content": user},
    ]
