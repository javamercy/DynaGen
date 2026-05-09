from dynagen.candidates.candidate import Candidate


BBOB_SOLVER_CONTRACT = """
Implement a complete continuous black-box optimizer with exactly this interface:

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        ...

    def __call__(self, func):
        ...

Functional rules:
- func(x) evaluates one NumPy-compatible point x with shape (dim,).
- The objective is minimization.
- The search bounds are available as func.bounds.lb and func.bounds.ub.
- Return the best observed objective value and its point as (best_value, best_x).
- Never call func more than self.budget times.
- Use self.dim and the bounds; do not assume a fixed dimension.
- seed must control stochastic behavior when randomness is used.
- Always create and evaluate at least one feasible incumbent early.
- Keep all candidate points inside the bounds, using clipping or bounded sampling.
- Do not hard-code behavior for specific BBOB function ids, instance ids, seeds, or evaluator artifacts.
- Do not read files, write files, access the network, spawn subprocesses, or call external solvers.
- Allowed imports only: numpy, math, random, heapq, itertools, collections, time.

Timeout and reporting rules:
- A global helper report_best(value, x) is available.
- Call report_best(best_value, best_x) whenever you improve the incumbent.
- report_best does not replace returning (best_value, best_x).

Implementation rules:
- The submitted code may include helper functions and the Optimizer class.
- Keep the implementation compact and robust.
- Prefer bounded, budget-aware search loops over unbounded while loops.
- Avoid recursion-heavy or memory-heavy designs.
"""


BBOB_INTERNAL_CHECKLIST = """
Before producing the final response, internally verify the candidate:

1. The code defines class Optimizer with __init__(self, budget, dim, seed) and __call__(self, func).
2. The optimizer evaluates at least one feasible point before any expensive logic.
3. Every func call is counted or guarded so total calls never exceed self.budget.
4. All points are finite and clipped or sampled within func.bounds.lb and func.bounds.ub.
5. report_best(best_value, best_x) is called for the initial incumbent and every improvement.
6. seed controls stochastic behavior if randomness is used.
7. Only allowed imports are used.
8. The code does not read files, write files, access the network, spawn subprocesses, or call external solvers.
9. The implementation avoids obvious infinite loops and unguarded expensive repeated work.
"""


BBOB_RESPONSE_FORMAT = """
Return exactly one JSON object and nothing else.

Schema:
{
  "name": "short snake_case_or_title name",
  "thought": "brief public summary of the optimization idea, including initialization, improvement, diversification, seed use, and budget use",
  "code": "complete Python code as a JSON string"
}

Strict formatting rules:
- Do not use Markdown.
- Do not include ``` fences.
- Do not include comments outside the JSON.
- The code field must be a valid JSON string.
- Escape newlines in code as \\n.
- Escape inner double quotes as needed.
- The code must define class Optimizer with the required methods.
"""


def bbob_system_prompt(role: str) -> str:
    return f"""
You are {role}. Generate robust, compact Python optimizers for continuous black-box minimization.
Focus on full optimizers, not small heuristic components.
Use classical continuous optimization ideas when useful, but combine them into a coherent complete optimizer.
Use budget as a hard function-evaluation cap. Follow the contract exactly."""


def render_bbob_candidates(candidates: list[Candidate]) -> str:
    return "\n\n".join(_render_bbob_candidate(candidate) for candidate in candidates)


def _render_bbob_candidate(candidate: Candidate) -> str:
    fitness = "unknown" if candidate.fitness is None else f"{candidate.fitness:.6g}"
    metrics = candidate.metrics or {}
    parts = [
        f"Candidate {candidate.id}: {candidate.name}",
        f"Status: {candidate.status}; fitness: {fitness}",
        f"Thought: {candidate.thought}",
        f"Mean AOCC: {metrics.get('mean_aocc')}",
        f"Mean final error: {metrics.get('mean_final_error')}",
        f"AOCC by group: {metrics.get('aocc_by_group')}",
        "Code:",
        "```python",
        candidate.code,
        "```",
    ]
    if candidate.error_details:
        parts.insert(6, f"Error details: {candidate.error_details}")
    return "\n".join(parts)
