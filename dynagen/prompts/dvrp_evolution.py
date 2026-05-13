from dynagen.candidates.candidate import Candidate
from dynagen.prompts.dvrp_templates import (
    DVRP_INTERNAL_CHECKLIST,
    DVRP_POLICY_CONTRACT,
    DVRP_RESPONSE_FORMAT,
    render_dvrp_candidates,
)

DVRP_STRATEGY_INSTRUCTIONS = {
    "S1": """Explore: create a complete dispatch policy with materially different decision behavior from the parent.
Change a core decision mechanism, not names or constants. Keep validity, budget use, and deterministic seed handling.""",

    "S2": """Refine: use parent metrics and reflection to make one or two targeted fixes.
Preserve what works, address measured weakness, and avoid unrelated rewrites.""",

    "S3": """Recombine: build one coherent policy from complementary parent strengths.
Do not concatenate parents or run them sequentially. Resolve conflicts and keep the child simpler than the sum.""",
}

def build_dvrp_evolution_prompt(
        strategy: str,
        parents: list[Candidate],
        *,
        generation_reflection: str = "",
) -> list[dict[str, str]]:
    if strategy not in DVRP_STRATEGY_INSTRUCTIONS:
        raise ValueError(f"Unknown strategy: {strategy}")
    blocks = [
        f"STRATEGY {strategy}: {DVRP_STRATEGY_INSTRUCTIONS[strategy]}",
        "Minimize time until the last truck returns to the depot. This is the only optimization goal.",
        "When customers are available, prefer assigning one instead of waiting unless waiting is clearly better.",
    ]
    if generation_reflection:
        blocks.append(f"REFLECTION FROM RECENT PARENT/CHILD COMPARISON:\n{generation_reflection}")
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
