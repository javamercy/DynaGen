from dynagen.candidates.candidate import Candidate
from dynagen.evolution.history import format_history_parent_context

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
- Keep all candidate points inside the bounds, using clipping or bounded sampling.
- Do not hard-code behavior for specific BBOB function ids, instance ids, seeds, or evaluator artifacts.
- Do not read files, write files, access the network, spawn subprocesses, or call external solvers.
- Allowed imports only: numpy, math, random, heapq, itertools, collections, time, numba.
- Numba may be used for small hot numeric helper functions only; keep Optimizer methods as Python wrappers around bounded search logic, and do not use Numba caching or object-mode patterns.

Timeout and reporting rules:
- A global helper report_best(value, x) is available.
- Call report_best(best_value, best_x) whenever you improve the incumbent.
- report_best does not replace returning (best_value, best_x).
"""

BBOB_INTERNAL_CHECKLIST = """
Before producing the final response, internally verify the candidate:

1. The code defines class Optimizer with __init__(self, budget, dim, seed) and __call__(self, func).
2. The optimizer evaluates at least one feasible point before any expensive logic.
3. Every func call is counted or guarded so total calls never exceed self.budget.
4. All points are finite and clipped or sampled within func.bounds.lb and func.bounds.ub.
5. report_best(best_value, best_x) is called for the initial incumbent and every improvement.
6. seed controls stochastic behavior if randomness is used.
7. Only allowed imports are used; Numba helpers, if any, are limited to hot numeric kernels.
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


def bbob_system_prompt() -> str:
    return f"""
        You generate robust, compact Python optimizers for continuous black-box minimization.
        DynaGen ranks BBOB candidates by mean AOCC; higher mean AOCC is better.
        Follow the requested interface exactly.
        """


def render_bbob_candidates(candidates: list[Candidate]) -> str:
    return "\n\n".join(_render_bbob_candidate(candidate) for candidate in candidates)


def _render_bbob_candidate(candidate: Candidate) -> str:
    metrics = candidate.metrics or {}
    score = candidate.score_value if candidate.score_name == "mean_aocc" else metrics.get("mean_aocc")
    mean_aocc_score = "unknown" if score is None else f"{float(score):.6g}"
    parts = [
        f"Candidate {candidate.id}: {candidate.name}",
        f"Status: {candidate.status}; mean AOCC score: {mean_aocc_score} (higher is better)",
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
    history_context = format_history_parent_context(candidate)
    if history_context:
        parts.insert(6 if not candidate.error_details else 7, history_context)
    return "\n".join(parts)
