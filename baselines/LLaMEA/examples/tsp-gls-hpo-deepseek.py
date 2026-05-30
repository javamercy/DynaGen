"""
LLaMEA-HPO for Traveling Salesman Problem (TSP) with Guided Local Search
Using DeepSeek Model - STANDALONE (No external dependencies except listed in pyproject.toml)

This example demonstrates how to use LLaMEA with SMAC3 for hyperparameter optimization
of TSP heuristics. The framework generates novel edge distance update strategies
that help escape local optima in Guided Local Search.

Paper: "In-the-Loop Automated Algorithm Design for Algorithms Generalization"
Model: deepseek-v4-flash (cost-effective alternative to GPT-4o)
Problem: TSP with 10 instances
HPO: SMAC3-based hyperparameter optimization (50 evaluations per iteration)
Iterations: 20

How to run:
  cd baselines/LLaMEA
  uv sync --dev
  export DEEPSEEK_API_KEY="sk-..."
  uv run python examples/tsp-gls-hpo-deepseek.py
"""

import os
import sys
import time
import textwrap
import logging
import numpy as np
from datetime import datetime
from math import sqrt
from ConfigSpace import Configuration, ConfigurationSpace
from smac import AlgorithmConfigurationFacade, Scenario

from llamea import DeepSeek_LLM, LLaMEA

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Standalone TSPLIB Parser (No external dependencies)
# ============================================================================

def parse_tsplib(content: str):
    """Parse TSPLIB TSP file format - standalone implementation."""
    lines = content.strip().split('\n')

    data = {
        'name': None,
        'dimension': None,
        'optimal': None,
        'coordinates': [],
    }

    section = None

    for line in lines:
        line = line.strip()
        if not line or line.upper() == 'EOF':
            continue

        # Check for section headers (no colon)
        if line.upper() in ['NODE_COORD_SECTION', 'EDGE_WEIGHT_SECTION', 'DISPLAY_DATA_SECTION']:
            section = line.upper()
            continue

        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()

            if key == 'NAME':
                data['name'] = value
            elif key == 'DIMENSION':
                data['dimension'] = int(value)
            elif key == 'OPTIMAL':
                try:
                    data['optimal'] = float(value)
                except:
                    pass
        else:
            if section == 'NODE_COORD_SECTION':
                parts = line.split()
                if len(parts) >= 3:
                    idx, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                    data['coordinates'].append((x, y))

    return data


def load_tsp_instance(filepath: str):
    """Load TSP instance from file and compute distance matrix."""
    with open(filepath, 'r') as f:
        content = f.read()

    data = parse_tsplib(content)

    n = data['dimension']
    coords = data['coordinates']

    # Compute Euclidean distance matrix
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist_matrix[i][j] = sqrt(dx*dx + dy*dy)

    return {
        'name': data['name'],
        'dimension': n,
        'distance_matrix': dist_matrix,
        'optimal_length': data['optimal'],
        'coordinates': coords,
    }


# ============================================================================
# GLS Implementation for TSP
# ============================================================================

class TSPGuidedLocalSearch:
    """
    Guided Local Search (GLS) for TSP that accepts custom distance update functions.
    This allows LLM-generated heuristics to dynamically penalize frequently used edges.
    """

    @staticmethod
    def nearest_neighbor(distance_matrix, start=0):
        """Initialize tour using nearest neighbor heuristic."""
        n = distance_matrix.shape[0]
        tour = np.zeros(n, dtype=np.uint16)
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour[0] = start

        for i in range(1, n):
            min_dist = np.inf
            min_idx = -1
            for j in range(n):
                if not visited[j] and distance_matrix[tour[i-1], j] < min_dist:
                    min_dist = distance_matrix[tour[i-1], j]
                    min_idx = j
            tour[i] = min_idx
            visited[min_idx] = True

        return tour

    @staticmethod
    def tour_cost(distance_matrix, tour):
        """Calculate total tour cost."""
        cost = distance_matrix[tour[-1], tour[0]]
        for i in range(len(tour) - 1):
            cost += distance_matrix[tour[i], tour[i+1]]
        return cost

    @staticmethod
    def two_opt_move(distance_matrix, tour):
        """Perform one pass of 2-opt."""
        n = len(tour)
        improvement = 0.0
        best_delta = 0.0
        best_i, best_j = 0, 0

        for i in range(1, n - 1):
            for j in range(i + 1, n):
                node_i = tour[i]
                node_j = tour[j]
                node_prev = tour[i-1]
                node_next = tour[(j+1) % n]

                if node_prev == node_j or node_next == node_i:
                    continue

                delta = (distance_matrix[node_prev, node_j] +
                        distance_matrix[node_i, node_next] -
                        distance_matrix[node_prev, node_i] -
                        distance_matrix[node_j, node_next])

                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j

        if best_delta < -1e-6:
            tour[best_i: best_j+1] = np.flip(tour[best_i: best_j+1])
            improvement = best_delta

        return improvement

    @staticmethod
    def local_search(distance_matrix, tour, max_iterations=1000):
        """Perform local search until convergence."""
        for iteration in range(max_iterations):
            improvement = TSPGuidedLocalSearch.two_opt_move(distance_matrix, tour)
            if improvement < -1e-6:
                continue
            else:
                break
        return 0.0

    @staticmethod
    def gls(distance_matrix, update_distance_fn, gls_iters=100, ls_iters=1000):
        """
        Run GLS with custom distance update function.

        Args:
            distance_matrix: Initial distance matrix
            update_distance_fn: Function that takes (dist, tour, edge_count) and returns updated dist
            gls_iters: Number of GLS iterations
            ls_iters: Maximum local search iterations

        Returns:
            Best tour found and its cost (on original distance matrix)
        """
        n = distance_matrix.shape[0]

        # Initialize
        best_tour = TSPGuidedLocalSearch.nearest_neighbor(distance_matrix, start=0)
        TSPGuidedLocalSearch.local_search(distance_matrix, best_tour, ls_iters)
        best_cost = TSPGuidedLocalSearch.tour_cost(distance_matrix, best_tour)

        edge_count = np.zeros_like(distance_matrix, dtype=np.int_)
        current_dist = distance_matrix.copy()
        current_tour = best_tour.copy()

        for gls_iter in range(gls_iters):
            # Update edge usage
            for i in range(len(current_tour)):
                u = current_tour[i]
                v = current_tour[(i + 1) % len(current_tour)]
                edge_count[u, v] += 1
                edge_count[v, u] += 1

            # Apply LLM-generated distance update
            try:
                current_dist = update_distance_fn(
                    current_dist.copy(),
                    current_tour.copy(),
                    edge_count.copy()
                )
            except Exception as e:
                logger.debug(f"Error in update_distance at iteration {gls_iter}: {e}")
                break

            # Local search
            search_tour = current_tour.copy()
            TSPGuidedLocalSearch.local_search(current_dist, search_tour, ls_iters)
            search_cost = TSPGuidedLocalSearch.tour_cost(distance_matrix, search_tour)

            # Track best
            if search_cost < best_cost:
                best_tour = search_tour.copy()
                best_cost = search_cost
                current_tour = search_tour.copy()
            else:
                current_tour = search_tour.copy()

        return best_tour, best_cost


# ============================================================================
# TSP Instance Loading (Standalone - no external dependencies)
# ============================================================================

def load_tsp_instances(base_dir="../data/tsp/test_instances"):
    """Load the 10 TSP test instances.

    Args:
        base_dir: Path to TSP instances directory (relative to examples directory)
    """
    instance_files = [
        "pr439.tsp", "fl417.tsp", "d198.tsp", "u724.tsp", "a280.tsp",
        "d493.tsp", "p654.tsp", "rat783.tsp", "lin318.tsp", "u574.tsp",
    ]

    instances = {}
    for filename in instance_files:
        # Handle both relative and absolute paths
        if os.path.isabs(base_dir):
            filepath = os.path.join(base_dir, filename)
        else:
            # Relative to the examples directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, base_dir, filename)

        if not os.path.exists(filepath):
            logger.warning(f"Instance not found: {filepath}")
            continue

        try:
            instance_data = load_tsp_instance(filepath)
            instances[instance_data['name']] = {
                "distance_matrix": instance_data['distance_matrix'],
                "optimal_length": instance_data['optimal_length'],
                "dimension": instance_data['dimension'],
            }
            logger.info(f"✓ Loaded {instance_data['name']}: {instance_data['dimension']} nodes, optimal={instance_data['optimal_length']}")
        except Exception as e:
            logger.error(f"✗ Failed to load {filename}: {e}")

    if not instances:
        raise ValueError(f"No TSP instances loaded from {base_dir}. "
                        f"Make sure TSP files are in the correct location.")

    return instances


# ============================================================================
# Evaluation with SMAC3 HPO
# ============================================================================

def compute_gap(tour_length, optimal_length):
    """Compute gap percentage from optimal."""
    if optimal_length is None or optimal_length <= 0:
        return np.inf
    if tour_length <= 0 or not np.isfinite(tour_length):
        return np.inf
    return 100.0 * (tour_length - optimal_length) / optimal_length


def evaluate_tsp_heuristic(solution, instances):
    """
    Evaluate an LLM-generated heuristic with SMAC3 HPO.

    Steps:
    1. Execute the generated heuristic code
    2. Use SMAC3 to optimize hyperparameters on all instances
    3. Evaluate final performance
    4. Return fitness (mean gap)
    """
    code = solution.code
    heuristic_name = solution.name

    logger.info(f"Evaluating: {heuristic_name}")

    # Execute code
    try:
        exec(code, globals())
    except Exception as e:
        logger.error(f"Execution error: {e}")
        solution.set_scores(np.inf, f"Execution error: {e}", error=e)
        return solution

    if heuristic_name not in globals():
        logger.error(f"Class {heuristic_name} not found in generated code")
        solution.set_scores(np.inf, f"Class {heuristic_name} not found")
        return solution

    # Quick validation
    try:
        logger.info("Running validation...")
        instance_name = list(instances.keys())[0]
        instance = instances[instance_name]
        heuristic = globals()[heuristic_name]()
        tour, cost = TSPGuidedLocalSearch.gls(
            instance["distance_matrix"],
            heuristic.update_edge_distance,
            gls_iters=5
        )
        logger.info(f"Validation OK on {instance_name}: cost={cost:.1f}")
    except Exception as e:
        logger.error(f"Validation error: {e}")
        solution.set_scores(np.inf, f"Validation error: {e}", error=e)
        return solution

    # SMAC3 HPO
    logger.info("Running SMAC3 hyperparameter optimization (50 evaluations)...")

    if solution.configspace is None:
        incumbent = {}
        logger.info("No configuration space - using default hyperparameters")
    else:
        try:
            configuration_space = solution.configspace

            def evaluate_config(config: Configuration, seed: int = 0):
                """Objective function for SMAC3 - evaluate on all instances."""
                np.random.seed(seed)
                gaps = []
                for instance_name, instance in instances.items():
                    try:
                        heuristic = globals()[heuristic_name](**dict(config))
                        tour, cost = TSPGuidedLocalSearch.gls(
                            instance["distance_matrix"],
                            heuristic.update_edge_distance,
                            gls_iters=20
                        )
                        gap = compute_gap(cost, instance["optimal_length"])
                        gaps.append(gap if np.isfinite(gap) else 1000.0)
                    except Exception as e:
                        logger.debug(f"Error on {instance_name}: {e}")
                        gaps.append(1000.0)
                return np.mean(gaps) if gaps else 1000.0

            scenario = Scenario(
                configuration_space,
                name=f"tsp-{int(time.time())}-{heuristic_name}",
                deterministic=False,
                n_trials=50,
                output_directory="smac3_output"
            )

            smac = AlgorithmConfigurationFacade(scenario, evaluate_config, logging_level=30)
            incumbent = smac.optimize()
            logger.info(f"SMAC3 completed. Best config: {dict(incumbent)}")

        except Exception as e:
            logger.error(f"SMAC3 error: {e}")
            incumbent = {}

    # Final evaluation
    logger.info("Final evaluation on all instances...")
    gaps = []
    for instance_name, instance in instances.items():
        try:
            heuristic = globals()[heuristic_name](**incumbent)
            tour, cost = TSPGuidedLocalSearch.gls(
                instance["distance_matrix"],
                heuristic.update_edge_distance,
                gls_iters=50
            )
            gap = compute_gap(cost, instance["optimal_length"])
            gaps.append(gap if np.isfinite(gap) else 1000.0)
            logger.info(f"  {instance_name}: gap={gap:.2f}%")
        except Exception as e:
            gaps.append(1000.0)
            logger.error(f"  {instance_name}: {e}")

    mean_gap = np.mean(gaps) if gaps else np.inf
    std_gap = np.std(gaps) if len(gaps) > 1 else 0.0

    solution.add_metadata("gaps", gaps)
    solution.add_metadata("mean_gap", mean_gap)
    solution.add_metadata("config", dict(incumbent))

    feedback = f"Mean gap: {mean_gap:.2f}% ± {std_gap:.2f}% | Config: {dict(incumbent)}"
    logger.info(f"Result: {feedback}")

    solution.set_scores(mean_gap, feedback)
    return solution


# ============================================================================
# Main LLaMEA Configuration with DeepSeek
# ============================================================================

if __name__ == "__main__":
    # API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY environment variable not set")
        logger.info("Please set your DeepSeek API key before running:")
        logger.info("  export DEEPSEEK_API_KEY='sk-...'")
        logger.info("Then run this script again from the LLaMEA root directory:")
        logger.info("  cd baselines/LLaMEA")
        logger.info("  uv run python examples/tsp-gls-hpo-deepseek.py")
        sys.exit(1)

    logger.info("="*70)
    logger.info("LLaMEA-HPO for TSP (DeepSeek Version - Standalone)")
    logger.info("="*70)

    # LLM Configuration with DeepSeek
    logger.info("Initializing LLM: deepseek-v4-flash")
    start_time = time.time()
    llm = DeepSeek_LLM(api_key, model="deepseek-v4-flash")
    logger.info(f"LLM initialized (took {time.time() - start_time:.2f}s)")

    # Load instances
    logger.info("Loading TSP instances...")
    load_start = time.time()
    instances = load_tsp_instances()
    if not instances:
        logger.error("No TSP instances loaded")
        sys.exit(1)

    logger.info(f"Loaded {len(instances)} TSP instances (took {time.time() - load_start:.2f}s)")

    # LLaMEA Configuration
    experiment_name = f"tsp-gls-hpo-deepseek-{datetime.now().strftime('%m-%d_%H%M%S')}"
    n_gens = 20

    # Task prompt from the LLaMEA-HPO paper
    task_prompt = textwrap.dedent("""
    Task: Given an edge distance matrix and a local optimal route, please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance (TSP problem).
    You should create an algorithm for me to update the edge distance matrix. Provide the Python code for the new strategy. The code is a Python class that should contain two functions an '__init__()' function containing any hyper-parameters that can be optimized, and a function called 'update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used)' that takes three inputs, and outputs the 'updated_edge_distance', where 'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrices, 'edge_n_used' includes the number of each edge used during permutation. All are Numpy arrays. The novel function should be sufficiently complex in order to achieve better performance. It is important to ensure self-consistency.
    An example heuristic to show the structure is as follows.
    ```python
    import numpy as np
    class Sample:
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

        def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
            # code here
            return updated_edge_distance
    ```

    In addition, any hyper-parameters the algorithm used will be optimized by SMAC, for this, provide a Configuration space as Python dictionary (without the edge_distance, local_opt_tour, edge_n_used parameters) and include all hyper-parameters to be optimized in the __init__ function header.
    An example configuration space is as follows:
    ```python
    {
        "float_parameter": (0.1, 1.5),
        "int_parameter": (2, 10),
        "categoral_parameter": ["mouse", "cat", "dog"]
    }
    ```

    Give an excellent and novel heuristic including its configuration space to solve this task and also give it a name.
    """)

    format_prompt = textwrap.dedent("""
    Give the response in the format:
    # Name: <name>
    # Code: <code>
    # Space: <configuration_space>
    """)

    example_prompt = textwrap.dedent("""
    An example of a simple penalty-based heuristic:
    ```python
    import numpy as np

    class PenaltyUpdate:
        def __init__(self, alpha=0.5, decay=0.95):
            self.alpha = alpha
            self.decay = decay

        def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
            penalty = self.alpha * edge_n_used
            updated = edge_distance + penalty
            return updated
    ```
    """)

    feedback_prompts = [
        "Refine the heuristic to improve gap performance (give it a new name).",
        "Redesign with more sophisticated penalty mechanisms.",
    ]

    # Create evaluation function with correct signature (solution, logger)
    def eval_func(solution, llamea_logger):
        return evaluate_tsp_heuristic(solution, instances)

    # Initialize and run LLaMEA with DeepSeek
    logger.info(f"Initializing LLaMEA with {n_gens} iterations")
    es = LLaMEA(
        eval_func,
        llm=llm,
        role_prompt="You are an expert in combinatorial optimization and metaheuristic design.",
        task_prompt=task_prompt,
        example_prompt=example_prompt,
        output_format_prompt=format_prompt,
        mutation_prompts=feedback_prompts,
        experiment_name=experiment_name,
        elitism=True,
        HPO=True,
        budget=n_gens,
        eval_timeout=3600,
        parallel_backend="threading",
        minimization=True,
    )

    logger.info("="*70)
    logger.info(f"Starting LLaMEA-HPO optimization")
    logger.info(f"Model: deepseek-v4-flash")
    logger.info(f"Iterations: {n_gens}")
    logger.info(f"Instances: {len(instances)}")
    logger.info(f"HPO Budget: 50 evaluations per iteration")
    logger.info(f"Experiment: {experiment_name}")
    logger.info("="*70)

    optimization_start = time.time()
    result = es.run()
    optimization_time = time.time() - optimization_start

    logger.info("="*70)
    logger.info(f"LLaMEA-HPO optimization complete")
    logger.info(f"Total time: {optimization_time/60:.1f} minutes ({optimization_time:.0f}s)")
    logger.info(f"Best result: {result}")
    logger.info("="*70)
