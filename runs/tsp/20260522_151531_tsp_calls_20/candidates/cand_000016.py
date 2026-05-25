import numpy as np
import random
import math

def report_best_tour(tour):
    pass

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.arange(n)
    if n == 2:
        return np.array([0, 1])

    random.seed(seed)
    np.random.seed(seed)

    def get_tour_dist(t):
        d = 0
        for i in range(n):
            d += distance_matrix[t[i], t[(i+1)%n]]
        return d

    # 1. Randomized Greedy Construction
    # Use a small candidate set to pick the next node to introduce randomness
    start_node = random.randint(0, n - 1)
    tour = [start_node]
    unvisited = set(range(n))
    unvisited.remove(start_node)

    while unvisited:
        curr = tour[-1]
        # Find nearest neighbors
        candidates = list(unvisited)
        if len(candidates) > 10:
            # Sort by distance and take top 5
            dists = [distance_matrix[curr, c] for c in candidates]
            sorted_indices = np.argsort(dists)
            top_candidates = [candidates[i] for i in sorted_indices[:5]]
            next_node = random.choice(top_candidates)
        else:
            # Just pick the closest among few
            dists = [distance_matrix[curr, c] for c in candidates]
            next_node = candidates[np.argmin(dists)]
        
        tour.append(next_node)
        unvisited.remove(next_node)

    current_tour = np.array(tour)
    current_dist = get_tour_dist(current_tour)
    best_tour = np.copy(current_tour)
    best_dist = current_dist
    report_best_tour(best_tour)

    # 2. Node Relocation Search (Materially different from 2-opt)
    # Instead of reversing segments, we pick a node and try to insert it elsewhere
    # This is a form of 3-opt (specifically, a node shift)
    iters = 0
    while iters < budget:
        # Randomly select a node to move
        node_idx = random.randint(0, n - 1)
        node = current_tour[node_idx]
        
        # Current neighbors
        prev_node = current_tour[(node_idx - 1) % n]
        next_node = current_tour[(node_idx + 1) % n]
        
        # Cost of removing 'node' from current position
        cost_remove = distance_matrix[prev_node, node] + distance_matrix[node, next_node] - distance_matrix[prev_node, next_node]
        
        # Try inserting 'node' at a different position
        # To keep it efficient, we sample potential insertion points
        target_idx = random.randint(0, n - 1)
        if target_idx == node_idx or (target_idx + 1) % n == node_idx:
            iters += 1
            continue
            
        # New neighbors at target position
        t_prev = current_tour[target_idx]
        t_next = current_tour[(target_idx + 1) % n]
        
        # Cost of inserting 'node' between t_prev and t_next
        cost_insert = distance_matrix[t_prev, node] + distance_matrix[node, t_next] - distance_matrix[t_prev, t_next]
        
        delta = cost_insert - cost_remove
        iters += 1

        if delta < -1e-9:
            # Perform relocation
            temp_tour = list(current_tour)
            val = temp_tour.pop(node_idx)
            # Adjust target_idx if it shifted due to pop
            adj_target = target_idx
            if node_idx < target_idx:
                adj_target -= 1
            # Insert after adj_target
            temp_tour.insert(adj_target + 1, val)
            
            current_tour = np.array(temp_tour)
            current_dist += delta
            
            if current_dist < best_dist:
                best_dist = current_dist
                best_tour = np.copy(current_tour)
                report_best_tour(best_tour)

    return best_tour