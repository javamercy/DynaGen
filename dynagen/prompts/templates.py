from dynagen.candidates.candidate import Candidate

SOLVER_CONTRACT = """
Implement a complete TSP solver with exactly this interface:

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:

Functional rules:
- distance_matrix is a square NumPy array of edge costs.
- Return a 1D NumPy array or list containing each node id exactly once.
- Do not repeat the starting node at the end; the evaluator closes the cycle.
- The solver must work across different TSP sizes.
- For n <= 2, return the trivial valid tour.
- Always create a valid incumbent tour early, before expensive search.
- Always return a valid tour, even when budget is very small.
- seed must control stochastic behavior when randomness is used.
- budget is a hard effort cap. Use it to limit loop counts, restarts, local-search passes, perturbations, and candidate evaluations.
- Do not hard-code behavior for specific benchmark instances, matrix sizes, seeds, or evaluator artifacts.
- Use the distance matrix directly. Do not assume coordinates are available.
- Do not read files, write files, access the network, spawn subprocesses, or call external solvers.
- Allowed imports only: numpy, math, random, heapq, itertools, collections, time.

Timeout and reporting rules:
- A global helper report_best_tour(tour) is available.
- Call report_best_tour(tour) whenever you improve the incumbent tour.
- report_best_tour accepts the same tour format as the final return value.
- report_best_tour does not replace returning a final tour.
- The code should report an initial valid incumbent before entering expensive loops.

Implementation rules:
- The submitted code may include helper functions.
- Keep the implementation compact and robust.
- Prefer deterministic repair over raising exceptions.
- Avoid recursion-heavy or memory-heavy designs.
- Avoid O(n^3) operations inside large repeated loops unless guarded by n or budget.
"""

INTERNAL_CHECKLIST = """
Before producing the final response, internally verify the candidate:

1. The code defines solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int).
2. Every return path returns a valid tour containing each node exactly once.
4. A valid incumbent tour is created before expensive search.
5. report_best_tour(tour) is called for the initial incumbent and every improved incumbent.
6. seed controls stochastic behavior if randomness is used.
7. budget bounds the main search effort.
8. Only allowed imports are used.
9. The code does not read files, write files, access the network, spawn subprocesses, or call external solvers.
10. The solver is complete, not just a helper function.
11. The implementation avoids obvious infinite loops and unguarded expensive repeated O(n^3) work.
"""

RESPONSE_FORMAT = """
Return exactly one JSON object and nothing else.

Schema:
{
  "name": "short snake_case_or_title name",
  "thought": "brief public summary of the algorithmic idea, including construction, improvement, diversification, seed use, and budget use",
  "code": "complete Python code as a JSON string"
}

Strict formatting rules:
- Do not use Markdown.
- Do not include ``` fences.
- Do not include comments outside the JSON.
- The code field must be a valid JSON string.
- Escape newlines in code as \\n.
- Escape inner double quotes as needed.
- The code must define solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int).
"""


def base_system_prompt(role: str) -> str:
    return f"""
You are {role}. Generate robust, compact Python TSP solver code.
Focus on full solvers, not small heuristic components.
Use classical TSP heuristics when useful, but combine them into a coherent complete solver rather than a generic template.
Use budget as a hard effort cap. Follow the contract exactly."""


def render_candidates(candidates: list[Candidate]) -> str:
    return "\n\n".join(_render_candidate(candidate) for candidate in candidates)


def _render_candidate(candidate: Candidate) -> str:
    fitness = "unknown" if candidate.fitness is None else f"{candidate.fitness:.6g}"
    metrics = candidate.metrics or {}
    parts = [
        f"Candidate {candidate.id}: {candidate.name}",
        f"Status: {candidate.status}; fitness: {fitness}%",
        f"Thought: {candidate.thought}",
        f"Mean runtime: {metrics.get('mean_runtime')}",
        "Code:",
        "```python",
        candidate.code,
        "```",
    ]

    if candidate.error_details:
        parts.insert(4, f"Error details: {candidate.error_details}")

    return "\n".join(parts)

# TODO: Update candidate feedback info, add more metrics such as reflection feedback
# metrics = {
#   "mean_runtime": ...,
#   "timeout_rate": ...,
#   "invalid_rate": ...,
#   "reported_best_rate": ...,
#   "mean_gap_by_size": ...,
#   "best_gap": ...,
#   "worst_gap": ...,
#   "observed_strengths": ...,
#   "observed_weaknesses": ...,
# }
