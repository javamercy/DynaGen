from dynagen.candidates.candidate import Candidate
from dynagen.evolution.history import format_history_parent_context

VRP_SOLVER_CONTRACT = """
Implement exactly this interface:

def solve_vrp(
    distance_matrix: np.ndarray,  # (n, n), depot is node 0
    truck_count: int,
) -> list[list[int]]:

Rules:
- VRP is a static multi-truck routing problem; the objective is to minimize the maximum route distance across trucks.
- Return exactly truck_count routes.
- Each route must start at depot node 0 and end at depot node 0.
- Every customer node 1..n-1 must appear exactly once across all routes.
- An unused truck route must be [0, 0].
- Use only distance_matrix; do not assume coordinates, instance size, truck count, or dataset details.
- Keep all search and improvement loops finite and bounded by instance size; never use open-ended loops.
- Call report_best_vrp(routes) whenever you find a better complete feasible route set, especially before expensive improvement loops.
- Do not read/write files, use network, spawn subprocesses, or call external solvers.
- Allowed imports only: numpy, math, random, heapq, itertools, collections, time, numba.
- Numba may be used for small hot numeric helper functions only; keep solve_vrp as the Python wrapper that builds valid routes and calls report_best_vrp, and do not use Numba caching or object-mode patterns.
- No module-level mutable globals; the solver may be called many times across instances.
"""

VRP_INTERNAL_CHECKLIST = """
Internal check before final JSON: correct solve_vrp signature, exactly truck_count routes,
every route starts and ends at 0, every customer appears exactly once, empty trucks use [0, 0],
finite instance-size-bounded loops, deterministic tie handling, report_best_vrp used for
incumbents, allowed imports only, Numba helpers only for hot numeric kernels,
no I/O/network/subprocesses.
"""

VRP_RESPONSE_FORMAT = """
Return one JSON object and nothing else:

{
  "name": "short snake_case_or_title name",
  "thought": "brief public summary of the construction, balancing, improvement, and tie-break logic",
  "code": "complete Python code as a JSON string"
}

No Markdown, fences, or text outside JSON. The code string must define solve_vrp.
"""


def vrp_system_prompt() -> str:
    return (
        f"You generate a compact VRP metaheuristic that builds complete "
        "multi-truck routes and minimizes the maximum route distance"
    )


def render_vrp_candidates(candidates: list[Candidate]) -> str:
    return "\n\n".join(_render_vrp_candidate(candidate) for candidate in candidates)


def _render_vrp_candidate(candidate: Candidate) -> str:
    gap = candidate.score_value
    gap_str = "unknown" if gap is None else f"{float(gap):.6g}"
    metrics = candidate.metrics or {}
    parts = [
        f"Candidate {candidate.id}: {candidate.name}",
        f"Status: {candidate.status}; gap: {gap_str}",
        f"Thought: {candidate.thought}",
        f"Mean gap: {metrics.get('mean_gap')}",
        f"Mean max route distance: {metrics.get('mean_max_route_distance')}",
        f"Mean total route distance: {metrics.get('mean_total_route_distance')}",
        f"Gap by instance size: {metrics.get('gap_by_instance_size')}",
        f"Gap by truck count: {metrics.get('gap_by_truck_count')}",
    ]
    if candidate.error_details:
        parts.append(f"Error details: {candidate.error_details}")
    history_context = format_history_parent_context(candidate)
    if history_context:
        parts.append(history_context)
    parts.extend([
        "Code:",
        "```python",
        candidate.code,
        "```",
    ])
    return "\n".join(parts)
