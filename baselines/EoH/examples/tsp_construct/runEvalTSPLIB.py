import os
import sys
import json
import glob
import importlib
import numpy as np

# Ensure parsers.tsp_parser can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from parsers.tsp_parser import get_tsp_dataset

def tour_cost(route, dist_matrix):
    cost = 0
    n = len(route)
    for j in range(n - 1):
        cost += dist_matrix[int(route[j]), int(route[j + 1])]
    cost += dist_matrix[int(route[-1]), int(route[0])]
    return cost

def generate_neighborhood_matrix(dist_matrix):
    n = len(dist_matrix)
    neighborhood_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        sorted_indices = np.argsort(dist_matrix[i])
        neighborhood_matrix[i] = sorted_indices
    return neighborhood_matrix

def evaluate_instance(instance_data, heuristic_func):
    coords = instance_data['coords']
    dist_matrix = instance_data['dist_matrix']
    problem_size = instance_data['dimension']
    neighbor_size = min(50, problem_size)

    neighbor_matrix = generate_neighborhood_matrix(dist_matrix)
    destination_node = 0
    current_node = 0
    route = np.zeros(problem_size, dtype=int)

    for i in range(1, problem_size - 1):
        near_nodes = neighbor_matrix[current_node][1:]
        mask = ~np.isin(near_nodes, route[:i])
        unvisited_near_nodes = near_nodes[mask]
        unvisited_near_size = np.minimum(neighbor_size, unvisited_near_nodes.size)
        unvisited_near_nodes = unvisited_near_nodes[:unvisited_near_size]

        next_node = heuristic_func(current_node, destination_node, unvisited_near_nodes, dist_matrix)
        current_node = next_node
        route[i] = current_node

    mask = ~np.isin(np.arange(problem_size), route[:problem_size - 1])
    last_node = np.arange(problem_size)[mask]
    current_node = last_node[0]
    route[problem_size - 1] = current_node

    cost = tour_cost(route, dist_matrix)
    return cost

def main():
    if len(sys.argv) < 2:
        print("Usage: python runEvalTSPLIB.py <results_dir_name>")
        sys.exit(1)

    results_dir_name = sys.argv[1]
    base_dir = os.path.dirname(__file__)
    target_dir = os.path.join(base_dir, results_dir_name)

    if not os.path.isdir(target_dir):
        print(f"Error: Directory {target_dir} not found.")
        sys.exit(1)

    pops_best_dir = os.path.join(target_dir, "results", "pops_best")
    json_files = glob.glob(os.path.join(pops_best_dir, "population_generation_*.json"))
    if not json_files:
        print(f"Error: No generation files found in {pops_best_dir}")
        sys.exit(1)

    # Find highest generation
    def extract_gen(f):
        base = os.path.basename(f)
        return int(base.split('_')[-1].split('.')[0])
        
    latest_file = max(json_files, key=extract_gen)
    print(f"Loading heuristic from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
        code = data['code']

    # Write code to a temporary heuristic file in the same directory
    heuristic_path = os.path.join(base_dir, "tsplib_heuristic.py")
    with open(heuristic_path, 'w') as f:
        f.write("import numpy as np\n\n")
        f.write(code)

    # Import the written heuristic
    sys.path.insert(0, base_dir)
    try:
        import tsplib_heuristic
        importlib.reload(tsplib_heuristic)
        heuristic_func = tsplib_heuristic.select_next_node
    except Exception as e:
        print(f"Failed to import heuristic: {e}")
        if os.path.exists(heuristic_path):
            os.remove(heuristic_path)
        sys.exit(1)

    tsplib_dir = os.path.join(base_dir, '..', '..', '..', '..', 'data', 'tsp', 'test_instances')
    print(f"Loading TSPLIB instances from: {tsplib_dir}")
    dataset = get_tsp_dataset(tsplib_dir)

    print("\nEvaluating...")
    total_gap = 0
    valid_instances = 0
    results_str = "Instance\tSize\tOpt\tCost\tGap(%)\n"
    results_str += "-"*50 + "\n"

    for instance in dataset:
        name = instance['name']
        opt = instance['optimal']
        size = instance['dimension']
        
        if opt is None:
            print(f"Skipping {name} (no optimal value defined)")
            continue

        cost = evaluate_instance(instance, heuristic_func)
        gap = ((cost - opt) / opt) * 100
        total_gap += gap
        valid_instances += 1
        
        line = f"{name}\t{size}\t{opt}\t{cost}\t{gap:.2f}%"
        print(line)
        results_str += line + "\n"

    mean_gap = total_gap / valid_instances if valid_instances > 0 else 0
    summary = f"\nMean Optimality Gap: {mean_gap:.2f}% over {valid_instances} instances."
    print(summary)
    results_str += summary + "\n"

    # Save detailed evaluation inside the run's `results` directory
    inner_results_dir = os.path.join(target_dir, "results")
    os.makedirs(inner_results_dir, exist_ok=True)
    out_file = os.path.join(inner_results_dir, "eval_tsplib.txt")
    with open(out_file, 'w') as f:
        f.write(results_str)
    print(f"\nDetailed evaluation results saved to {out_file}")

    # Also append a summary line to a collective file in tsp_construct
    collective_eval_file = os.path.join(base_dir, "all_evals.txt")
    with open(collective_eval_file, 'a') as f:
        f.write(f"{results_dir_name}\tMean Gap: {mean_gap:.2f}%\n")
    print(f"Summary appended to {collective_eval_file}")

    # Clean up
    if os.path.exists(heuristic_path):
        os.remove(heuristic_path)

if __name__ == "__main__":
    main()
