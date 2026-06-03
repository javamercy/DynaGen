from dataclasses import dataclass

from dynagen.prompts.tsp_templates import (
    TSP_INTERNAL_CHECKLIST,
    TSP_RESPONSE_FORMAT,
    TSP_SOLVER_CONTRACT,
    tsp_system_prompt,
)


@dataclass(frozen=True)
class TSPInitialRole:
    slot: int
    role: str
    intended_bias: str


TSP_INITIAL_ROLES = [
    TSPInitialRole(
        slot=1,
        role="The Operations Research Specialist",
        intended_bias="You are an Operations Research Specialist. You approach the TSP purely through deterministic \n combinatorial optimization. Your focus is on rigorous local search heuristics (like 2-opt, 3-opt,\nLin-Kernighan), edge-exchange mechanisms, and greedy constructive algorithms with mathematical bounds."
        ),
    TSPInitialRole(
        slot=2,
        role="The Stochastic Statistician",
        intended_bias="You are a Stochastic Statistician. You view the TSP as a probability distribution and energy state \nproblem. Your strategies revolve around Simulated Annealing, Markov Chain Monte Carlo (MCMC), and \nprobabilistic acceptance criteria (like Metropolis-Hastings) to intelligently escape local optima."
        ),
    TSPInitialRole(
        slot=3,
        role="The High-Performance Software Engineer",
        intended_bias="You are a High-Performance Computing (HPC) Software Engineer. You care about raw execution speed, \nmemory locality, and algorithmic complexity. Your focus is on spatial partitioning (like KD-trees or \ngrid hashing), nearest-neighbor pre-computations, and ultra-fast cache-friendly array traversals using vectorized logic."
        ),
    TSPInitialRole(
        slot=4,
        role="The Nature-Inspired Researcher",
        intended_bias="You are a Nature-Inspired Metaheuristics Researcher. You design algorithms based on biological \nphenomena. Your focus is on Ant Colony Optimization (ACO) pheromone matrices, Genetic Algorithms \nwith specialized crossover operators (like PMX or OX), and swarm intelligence."
        ),
    TSPInitialRole(
        slot=5,
        role="The Graph Theory Mathematician",
        intended_bias="You are a Graph Theory Mathematician. You approach the TSP by analyzing graph properties. Your \nstrategies involve Minimum Spanning Trees (MST), Eulerian circuits, Christofides-inspired \napproximations, and manipulating node degrees or spanning forest topologies to build high-quality tours."
        ),
]


def build_tsp_initial_prompt(role: TSPInitialRole) -> list[dict[str, str]]:
    system = tsp_system_prompt()

    user = "\n\n".join([
        "# Initial Candidate Identity",
        f"Candidate ID: {role.slot}",
        f"Role: {role.role}",
        f"Intended bias: {role.intended_bias}",

        "# Internal Quality Checklist",
        TSP_INTERNAL_CHECKLIST.strip(),

        "# Solver Contract",
        TSP_SOLVER_CONTRACT.strip(),

        "# Response Format",
        TSP_RESPONSE_FORMAT.strip(),

    ])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
