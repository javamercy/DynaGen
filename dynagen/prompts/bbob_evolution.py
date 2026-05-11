from dynagen.candidates.candidate import Candidate
from dynagen.prompts.bbob_templates import (
    BBOB_INTERNAL_CHECKLIST,
    BBOB_RESPONSE_FORMAT,
    BBOB_SOLVER_CONTRACT,
    render_bbob_candidates,
)

BBOB_STRATEGY_INSTRUCTIONS = {
    "S1": """Explorative Innovation:
Generate a complete black-box optimizer that takes a meaningfully different algorithmic approach from the selected parent.

Do not make superficial changes. The offspring must differ in at least one core mechanism:
- the search distribution (e.g., Gaussian, Cauchy, uniform, correlated, coordinate-wise)
- step-size or covariance adaptation strategy (e.g., CSA, path-length control, self-adaptive σ)
- population structure or recombination scheme
- restart or diversification logic (e.g., IPOP/BIPOP-style restarts, random restarts with warm memory)
- hybrid structure (e.g., combining DE with local ES, or pattern search with population-based sampling)

Strong algorithmic families for BBOB include:
- CMA-ES variants with cumulative step-size adaptation and covariance learning
- Separable CMA-ES or per-coordinate step-size adaptation for separable functions
- Differential evolution with adaptive CR/F and optional local refinement
- Multi-restart strategies with increasing population size
- Hybrid optimizers that switch between global exploration and local exploitation based on progress

Use proven ideas, not novelty for its own sake.
Prioritize strict budget correctness, feasible incumbents, and strong anytime convergence.""",

    "S2": """Evidence-Guided Refinement:
Improve the selected parent into a stronger optimizer by diagnosing and fixing its weakest aspects.

Analyze the parent's AOCC scores by function group to identify specific weaknesses:
- Low separable AOCC → the optimizer may lack coordinate-wise adaptation or efficient axis-aligned search.
- Low high-conditioning AOCC → step-size control or covariance adaptation is likely too slow or missing.
- Low multimodal (strong structure) AOCC → restart logic may be absent, too infrequent, or too aggressive.
- Low multimodal (weak structure) AOCC → the optimizer may converge prematurely without sufficient global sampling.
- Low overall AOCC with high final error → the optimizer may waste evaluations on exploration without exploitation.

Make one or two coherent, targeted changes to address the diagnosed weakness.
Preserve mechanisms that contribute to the parent's strengths.
Do not make cosmetic changes. Every modification should have a clear performance rationale.

Also ensure robustness:
- feasible incumbent evaluated early, report_best called on every improvement
- budget never exceeded, all points clipped to bounds
- no infinite loops or unguarded expensive operations""",

    "S3": """Complementary Recombination:
Combine the strongest mechanisms from the selected parents into one coherent optimizer.

Do not concatenate or run parents sequentially. Choose one parent as the structural backbone,
then integrate one or two complementary mechanisms from the other parent.

Effective recombination targets:
- Take the better parent's core search loop, add the other's restart or diversification logic
- Combine one parent's step-size adaptation with the other's initialization or sampling strategy
- Merge a strong local refinement mechanism with a better global exploration strategy
- Integrate coordinate-wise adaptation from one parent with covariance learning from another

The offspring must be simpler than the sum of its parents.
Resolve design conflicts — do not include redundant or contradictory mechanisms.
The result should outperform both parents, not merely contain pieces of both.
Keep the implementation compact and budget-correct."""
}


def build_bbob_evolution_prompt(
        strategy: str,
        parents: list[Candidate],
) -> list[dict[str, str]]:
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
        {"role": "system",
         "content": "You generate executable, reliable continuous black-box optimizers for evolutionary search."},
        {"role": "user", "content": user},
    ]
