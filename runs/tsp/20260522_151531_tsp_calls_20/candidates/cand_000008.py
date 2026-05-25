import numpy as np
import random
import math

def report_best_tour(tour):
    # Placeholder for internal tracking
    pass

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 1:
        return np.arange(n)
    if n == 2:
        return np.array([0, 1])

    random.seed(seed)
    np.random.seed(seed)

    # 1. Initial Tour: Greedy Nearest Neighbor
    unvisited = set(range(1, n))
    current_tour = [0]
    curr_node = 0
    while unvisited:
        # Find nearest unvisited node
        next_node = min(unvisited, key=lambda x: distance_matrix[curr_node, x])
        current_tour.append(next_node)
        unvisited.remove(next_node)
        curr_node = next_node
    
    current_tour = np.array(current_tour)
    
    def get_total_dist(t):
        d = 0
        for i in range(n - 1):
            d += distance_matrix[t[i], t[i+1]]
        d += distance_matrix[t[-1], t[0]]
        return d

    current_dist = get_total_dist(current_tour)
    best_tour = np.copy(current_tour)
    best_dist = current_dist
    
    report_best_tour(best_tour)

    # 2. Simulated Annealing Parameters
    t_start = 100.0
    t_end = 0.01
    cooling_rate = math.pow(t_end / t_start, 1.0 / budget) if budget > 0 else 1.0
    temp = t_start

    # 3. Main Search Loop
    for i in range(budget):
        # 2-opt move: reverse a segment [idx1, idx2]
        idx1 = random.randint(0, n - 1)
        idx2 = random.randint(0, n - 1)
        if idx1 == idx2:
            continue
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        # Calculate delta for 2-opt swap
        prev_idx1 = (idx1 - 1) % n
        next_idx2 = (idx2 + 1) % n
        
        node_prev1 = current_tour[prev_idx1]
        node_idx1 = current_tour[idx1]
        node_idx2 = current_tour[idx2]
        node_next2 = current_tour[next_idx2]
        
        old_edges = distance_matrix[node_prev1, node_idx1] + distance_matrix[node_idx2, node_next2]
        new_edges = distance_matrix[node_prev1, node_idx2] + distance_matrix[node_idx1, node_next2]
        delta = new_edges - old_edges

        # Acceptance criteria
        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
            # Apply the swap (reverse the segment)
            current_tour[idx1 : idx2 + 1] = current_tour[idx1 : idx2 + 1][::-1]
            current_dist += delta
            
            if current_dist < best_dist:
                best_dist = current_dist
                best_tour = np.copy(current_tour)
                report_best_tour(best_tour)
        
        temp *= cooling_rate

    return best_tour