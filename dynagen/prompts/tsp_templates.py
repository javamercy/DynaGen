from dynagen.candidates.candidate import Candidate
from dynagen.evolution.verbal_gradient import format_candidate_verbal_gradient

TSP_SOLVER_CONTRACT = """
Implement exactly this interface:

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:

Rules:
- Distance is the search objective for TSP; lower distance is better.
- Return a 1D tour with each node exactly once; do not repeat the start node.
- Handle all n; for n <= 2 return the trivial valid tour.
- Always return a valid tour, even with tiny budget.
- Use only distance_matrix; do not assume coordinates or hard-code instances.
- Use seed for randomness and budget as a hard cap on search effort.
- Create a valid incumbent early; call report_best_tour(tour) for initial and improved incumbents.
- Do not read/write files, use network, spawn subprocesses, or call external solvers.
- Allowed imports only: numpy, math, random, heapq, itertools, collections, time.
- Keep code compact; avoid brittle repair, heavy recursion, and unguarded expensive loops.
"""

TSP_INTERNAL_CHECKLIST = """
Internal check before final JSON: correct signature, valid tour on every return path,
early report_best_tour, budget-bounded main search, allowed imports only, no I/O/network/subprocesses.
"""

TSP_RESPONSE_FORMAT = """
Return one JSON object and nothing else:

{
  "name": "short snake_case_or_title name",
  "thought": "brief public summary of the idea, seed use, and budget use",
  "code": "complete Python code as a JSON string"
}

No Markdown, fences, or text outside JSON. The code string must define solve_tsp.
"""


def tsp_system_prompt(role: str) -> str:
    return f"You are {role}. Generate compact TSP solver code that follows the contract."


def render_tsp_candidates(candidates: list[Candidate]) -> str:
    return "\n\n".join(
        _render_tsp_candidate(candidate)
        for candidate in candidates
    )


def _render_tsp_candidate(candidate: Candidate) -> str:
    distance = candidate.score_value
    distance_str = "unknown" if distance is None else f"{float(distance):.6g}"
    parts = [
        f"Candidate {candidate.id}: {candidate.name}",
        f"Status: {candidate.status}; distance: {distance_str}",
        f"Thought: {candidate.thought}",
    ]

    if candidate.error_details:
        parts.append(f"Error details: {candidate.error_details}")

    gradient = format_candidate_verbal_gradient(candidate)
    if gradient:
        parts.extend(["", gradient])

    parts.extend([
        "Code:",
        "```python",
        candidate.code,
        "```",
    ])
    return "\n".join(parts)
