from dynagen.candidates.candidate import Candidate
from dynagen.evolution.history import format_history_parent_context

TSP_SOLVER_CONTRACT = """
Implement this function only:

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:

Requirements:
- Minimize total TSP tour distance.
- Return a 1D np.ndarray containing each node exactly once.
- Do not repeat the start node at the end.
- Use only distance_matrix; do not assume coordinates.
- Create a valid tour early.
- Call report_best_tour(tour) whenever a new best valid tour is found, assume this func already exists.
- Do not use files, network, subprocesses, or external solvers.
- Keep the code compact and robust.
"""

TSP_INTERNAL_CHECKLIST = """
correct signature, valid tour on every return path,
early report_best_tour, allowed imports only, no I/O/network/subprocesses.
"""

TSP_RESPONSE_FORMAT = """
Return one JSON object and nothing else:

{
  "name": "short_snake_case_name",
  "thought": "high-level summary of the construction, improvement, and tie-break logic",
  "code": "complete Python code as a JSON string"
}
No Markdown, fences, or text outside JSON. The code string must define solve_tsp.
"""


def tsp_system_prompt() -> str:
    return """
    You generate compact, robust Python TSP metaheuristics.
    You must follow the requested interface exactly, always preserve tour validity.
    """


def render_tsp_candidates(candidates: list[Candidate]) -> str:
    return "\n\n".join(_render_tsp_candidate(candidate) for candidate in candidates)


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

    history_context = format_history_parent_context(candidate)
    if history_context:
        parts.append(history_context)

    parts.extend(
        [
            "Code:",
            "```python",
            candidate.code,
            "```",
        ]
    )
    return "\n\n".join(parts)
