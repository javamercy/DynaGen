from dynagen.candidates.candidate import Candidate

SOLVER_CONTRACT = """
Implement a complete TSP solver with exactly this interface:
def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:

Rules:
- distance_matrix is a square NumPy array.
- Return a 1D NumPy array or list containing each node id exactly once.
- Do not repeat the starting node at the end; the evaluator closes the cycle.
- A global helper report_best_tour(tour) is available. Call it whenever you improve your incumbent tour so the evaluator can score your best reported tour if the solver times out.
- report_best_tour accepts the same tour format as the final return value and does not replace returning a final tour.
- seed must control stochastic behavior! You don't have to use it if not needed.
- budget is a hard iteration/evaluation cap only; use it to limit compute, not to shape the solver's core design.
- Do not read files, write files, access the network, spawn subprocesses, or call external solvers.
- Allowed imports only: numpy, math, random, heapq, itertools, collections, time.
- The solver must work across different TSP sizes.
"""

RESPONSE_FORMAT = """
Return exactly JSON object below!! ONLY JSON OBJECT, NO EXPLANATION, NO MARKDOWN, NO TEXT:
{{
    "name": "a short name for the solver",
    "thought": "a brief explanation of the solver's main idea and how it uses the seed and budget",
    "code": "the complete code for the solver as a string, following the contract and guidance above"
}}
"""


def base_system_prompt(role: str) -> str:
    return f"""
You are {role}. Generate robust, compact Python TSP solver code.
Focus on full solvers, not small heuristic components. Avoid canonical TSP templates unless used only as minor repair logic.
Treat budget as an execution cap, not a design cue. Follow the contract exactly."""


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
