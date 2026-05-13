from dynagen.candidates.candidate import Candidate
from dynagen.evolution.verbal_gradient import format_candidate_verbal_gradient


DVRP_POLICY_CONTRACT = """
Implement exactly this interface:

def choose_next_customer(
    current_position: np.ndarray,        # (2,) the deciding truck's position
    depot_position: np.ndarray,          # (2,)
    truck_positions: np.ndarray,         # (n_trucks, 2) all trucks
    available_customers: np.ndarray,     # (n_available, 2) revealed unserved customers
    current_time: float,
    seed: int,
    budget: int,
) -> int | None:

Rules:
- DVRP is an online dispatch problem; the objective is to minimize the last-truck return time (makespan).
- Decide which customer the active truck (at current_position) should head to next.
- Return an index into available_customers, or None to wait at the current position.
- If available_customers is empty, return None.
- The function is stateless across calls; treat each call as a one-shot decision with the snapshot given.
- budget bounds compute for this single call; use it for internal lookahead, simulation, or scoring of alternatives if useful.
- Use seed for any randomness; ties must be deterministic.
- Do not assume coordinates beyond what is passed; do not hard-code instance sizes, truck counts, or dataset details.
- Do not read/write files, use network, spawn subprocesses, or call external solvers.
- Allowed imports only: numpy, math, random, heapq, itertools, collections, time.
- No module-level mutable globals; the function may be called many times across instances.
"""

DVRP_INTERNAL_CHECKLIST = """
Internal check before final JSON: correct choose_next_customer signature, returns None or a valid
available_customers index, handles empty available_customers, deterministic use of seed, respects
budget as a per-call compute cap, no module-level mutable state, allowed imports only, no
I/O/network/subprocesses.
"""

DVRP_RESPONSE_FORMAT = """
Return one JSON object and nothing else:

{
  "name": "short snake_case_or_title name",
  "thought": "brief public summary of the dispatch rule, tie-break, and wait condition",
  "code": "complete Python code as a JSON string"
}

No Markdown, fences, or text outside JSON. The code string must define choose_next_customer.
"""


def dvrp_system_prompt(role: str) -> str:
    return (
        f"You are {role}. Generate a compact online DVRP dispatch policy that "
        "minimizes the last-truck return time. The policy is stateless across "
        "decisions; do as much useful per-call reasoning as the budget allows."
    )


def render_dvrp_candidates(candidates: list[Candidate]) -> str:
    return "\n\n".join(_render_dvrp_candidate(candidate) for candidate in candidates)


def _render_dvrp_candidate(candidate: Candidate) -> str:
    distance = candidate.score_value
    distance_str = "unknown" if distance is None else f"{float(distance):.6g}"
    metrics = candidate.metrics or {}
    parts = [
        f"Candidate {candidate.id}: {candidate.name}",
        f"Status: {candidate.status}; distance: {distance_str}",
        f"Thought: {candidate.thought}",
        f"Mean gap: {metrics.get('mean_gap')}",
        f"Mean makespan: {metrics.get('mean_makespan')}",
        f"Gap by instance size: {metrics.get('score_by_instance_size')}",
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
