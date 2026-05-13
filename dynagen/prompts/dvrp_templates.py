from dynagen.candidates.candidate import Candidate
from dynagen.evolution.verbal_gradient import format_candidate_verbal_gradient


DVRP_POLICY_CONTRACT = """
Implement exactly this interface:

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
    seed: int,
    budget: int,
) -> int | None:

Rules:
- DVRP is an online dispatch problem; minimize time until the last truck returns to the depot.
- Choose the next customer using only the current online state.
- available_customers contains only currently known, unserved, unreserved customers.
- Return an index into available_customers, or None only when waiting is clearly better than serving any visible customer.
- If available_customers is empty, return None.
- Use current_position, depot_position, truck_positions, current_time, seed, and budget only.
- Use seed only for deterministic tie-breaking.
- Avoid waiting when customers are available; unnecessary waiting directly increases the last-truck finish time.
- Prefer short assignments, spatial partitioning between trucks, and choices that avoid making one truck finish much later than the rest.
- Keep each call cheap; this policy may be called hundreds of times in one simulation.
- Do not hard-code dataset sizes, seeds, truck counts, coordinates, or evaluator-specific details.
- Do not read/write files, use network, spawn subprocesses, or call external solvers.
- Allowed imports only: numpy, math, random, heapq, itertools, collections, time.
- Keep code compact and robust; avoid heavy recursion, global mutable state, and unbounded loops.
"""

DVRP_INTERNAL_CHECKLIST = """
Internal check before final JSON: correct choose_next_customer signature, returns None or a valid available_customers index,
handles empty and one-customer cases, avoids waiting when customers are available, minimizes last-truck return time,
uses only online state, has bounded per-call work, allowed imports only, no I/O/network/subprocesses.
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
    return f"You are {role}. Generate a compact online DVRP dispatch policy that minimizes the last-truck return time."


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
