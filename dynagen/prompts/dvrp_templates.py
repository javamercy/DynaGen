from dynagen.candidates.candidate import Candidate
from dynagen.evolution.history import format_history_parent_context

DVRP_POLICY_CONTRACT = """
Implement exactly this interface:

def choose_next_customer(
    current_position: np.ndarray,        # (2,) the deciding truck's position
    depot_position: np.ndarray,          # (2,)
    truck_positions: np.ndarray,         # (n_trucks, 2) all trucks
    available_customers: np.ndarray,     # (n_available, 2) revealed unserved customers
    current_time: float,
) -> int | None:

Rules:
- DVRP is an online dispatch problem; the objective is to minimize TTT, defined here as the last-truck return time.
- Decide which customer the active truck (at current_position) should head to next.
- Return an index into available_customers, or None to wait at the current position.
- If available_customers is empty, return None.
- The function is stateless across calls; treat each call as a one-shot decision with the snapshot given.
- Do not assume coordinates beyond what is passed; do not hard-code instance sizes, truck counts, or dataset details.
- Do not read/write files, use network, spawn subprocesses, or call external solvers.
- Allowed imports only: numpy, math, random, heapq, itertools, collections, time, numba.
- Numba may be used for small hot numeric helper functions only; keep choose_next_customer as the Python wrapper that returns a valid index or None, and do not use Numba caching or object-mode patterns.
- No module-level mutable globals; the function may be called many times across instances.
"""

DVRP_INTERNAL_CHECKLIST = """
Internal check before final JSON: correct choose_next_customer signature, returns None or a valid
available_customers index, handles empty available_customers, allowed imports only,
Numba helpers only for hot numeric kernels, no I/O/network/subprocesses.
"""

DVRP_RESPONSE_FORMAT = """
Return one JSON object and nothing else:
{
  "name": "snake_case_or_title",
  "thought": "brief public summary of the dispatch rule, tie-break, and wait condition",
  "code": "complete Python code as a JSON string"
}
"""


def dvrp_system_prompt(role: str) -> str:
    return (
        f"You are {role}. Generate a compact online DVRP dispatch policy that "
        "minimizes TTT (last-truck return time). The policy is stateless across "
        "decisions; do as much useful per-call reasoning as the budget allows."
    )


def render_dvrp_candidates(candidates: list[Candidate]) -> str:
    return "\n\n".join(_render_dvrp_candidate(candidate) for candidate in candidates)


def _render_dvrp_candidate(candidate: Candidate) -> str:
    gap = candidate.score_value
    gap_str = "unknown" if gap is None else f"{float(gap):.6g}%"
    metrics = candidate.metrics or {}
    parts = [
        f"Candidate {candidate.id}: {candidate.name}",
        f"Status: {candidate.status}; Gap: {gap_str}",
        f"Thought: {candidate.thought}",
        f"Mean gap: {metrics.get('mean_gap')}",
        f"Mean TTT: {metrics.get('mean_ttt')}",
        f"Gap by instance size: {metrics.get('score_by_instance_size')}",
        f"TTT by instance size: {metrics.get('ttt_by_instance_size')}",
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
