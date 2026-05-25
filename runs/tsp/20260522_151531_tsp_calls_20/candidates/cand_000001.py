import numpy as np
import random
import heapq

def report_best_tour(tour):
    # This is a placeholder for the internal tracking required by the prompt
    pass

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 1:
        return np.arange(n)
    if n == 2:
        return np.array([0, 1])

    random.seed(seed)
    np.random.seed(seed)

    # 1. Initial Greedy Incumbent
    unvisited = set(range(1, n))
    tour = [0]
    curr = 0
    while unvisited:
        next_node = min(unvisited, key=lambda x: distance_matrix[curr, x])
        tour.append(next_node)
        unvisited.remove(next_node)
        curr = next_node
    
    tour = np.array(tour)
    report_best_tour(tour)
    
    def get_dist(t):
        d = 0
        for i in range(n - 1):
            d += distance_matrix[t[i], t[i+1]]
        d += distance_matrix[t[-1], t[0]]
        return d

    best_dist = get_dist(tour)
    
    # 2. Candidate List Construction for scaling
    # For large n, we only consider the k nearest neighbors for each node
    k_neighbors = 20 if n >= 80 else n - 1
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        # Get indices of k smallest distances
        idx = np.argsort(dists)[:k_neighbors + 1]
        candidates.append(set(idx))

    # 3. Budget-bounded Local Search
    iterations = 0
    while iterations < budget:
        iterations += 1
        improved = False
        
        # Try 2-opt (edge swap)
        # To keep it fast, we pick a random edge and try swapping with its candidate neighbors
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        
        # Strict delta calculation for 2-opt: (i, i+1) and (j, j+1) -> (i, j) and (i+1, j+1)
        # Handle wrap-around for the last edge
        node_i = tour[i]
        node_ip1 = tour[(i + 1) % n]
        node_j = tour[j]
        node_jp1 = tour[(j + 1) % n]
        
        # Candidate check: only attempt if nodes are in each others' neighborhood or n is small
        if n < 80 or (node_i in candidates[node_j] or node_ip1 in candidates[node_jp1]):
            current_edges = distance_matrix[node_i, node_ip1] + distance_matrix[node_j, node_jp1]
            new_edges = distance_matrix[node_i, node_j] + distance_matrix[node_ip1, node_jp1]
            
            if new_edges < current_edges:
                # Perform 2-opt reverse
                tour[i+1 : j+1] = tour[i+1 : j+1][::-1]
                best_dist -= (current_edges - new_edges)
                report_best_tour(tour)
                improved = True
        
        if not improved:
            # Try Relocation move: move node i to position j
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j: continue
            
            # Extract node
            node = tour[i]
            temp_tour = np.delete(tour, i)
            temp_tour = np.insert(temp_tour, j, node)
            
            new_d = get_dist(temp_tour)
            if new_d < best_dist:
                tour = temp_tour
                best_dist = new_d
                report_best_tour(tour)
                improved = True

    return tour