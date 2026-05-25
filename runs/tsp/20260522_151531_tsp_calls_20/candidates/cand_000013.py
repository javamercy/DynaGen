import numpy as np
import random
import math

def report_best_tour(tour):
    pass

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 1:
        return np.arange(n)
    if n == 2:
        return np.array([0, 1])

    random.seed(seed)
    np.random.seed(seed)

    # 1. Greedy Initial Construction
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

    def get_total_dist(t):
        d = 0
        for i in range(n - 1):
            d += distance_matrix[t[i], t[i+1]]
        d += distance_matrix[t[-1], t[0]]
        return d

    best_tour = np.copy(tour)
    best_dist = get_total_dist(tour)
    curr_dist = best_dist

    # 2. Candidate List Construction
    k_neighbors = 20 if n >= 50 else n - 1
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        idx = np.argsort(dists)[:k_neighbors + 1]
        candidates.append(idx)

    # 3. Simulated Annealing with Candidate-Driven 2-Opt
    # Budget management: divide budget into iterations
    iterations = 0
    T = 100.0
    cooling_rate = 0.9995
    
    while iterations < budget:
        # Pick a random node and a random candidate neighbor
        u = random.randint(0, n - 1)
        w = random.choice(candidates[u])
        
        if w == u:
            continue

        # Find positions in current tour
        pos_u = np.where(tour == u)[0][0]
        pos_w = np.where(tour == w)[0][0]
        
        # Define edges for 2-opt
        i = pos_u
        j = pos_w
        if i > j: i, j = j, i
        
        # Edges are (i, i+1) and (j, j+1)
        node_i = tour[i]
        node_ip1 = tour[(i + 1) % n]
        node_j = tour[j]
        node_jp1 = tour[(j + 1) % n]
        
        # Delta calculation for 2-opt swap
        # Remove: (i, i+1) and (j, j+1) | Add: (i, j) and (i+1, j+1)
        # Note: if j = i+1, it's a degenerate swap
        if (i + 1) % n == j:
            # Special case: adjacent nodes, just a simple swap or no change
            # For simplicity in SA, we can skip or treat as 2-opt
            delta = 0
        else:
            current_edges = distance_matrix[node_i, node_ip1] + distance_matrix[node_j, node_jp1]
            new_edges = distance_matrix[node_i, node_j] + distance_matrix[node_ip1, node_jp1]
            delta = new_edges - current_edges

        iterations += 1
        
        # Acceptance criteria
        if delta < 0 or random.random() < math.exp(-delta / (T + 1e-9)):
            # Perform 2-opt reversal
            # Reverse segment from i+1 to j
            tour[i+1 : j+1] = tour[i+1 : j+1][::-1]
            curr_dist += delta
            
            if curr_dist < best_dist:
                best_dist = curr_dist
                best_tour = np.copy(tour)
                report_best_tour(best_tour)
        
        T *= cooling_rate
        if T < 0.01: T = 0.01 # Floor temperature

    return best_tour