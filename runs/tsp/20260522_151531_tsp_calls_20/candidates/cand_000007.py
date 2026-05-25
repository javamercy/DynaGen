import numpy as np
import random

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

    # 1. Initial Greedy Construction
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
    
    # 2. Candidate List Construction
    # Only consider k-nearest neighbors to keep search efficient
    k_neighbors = 25 if n >= 80 else n - 1
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        idx = np.argsort(dists)[:k_neighbors + 1]
        candidates.append(idx)

    # 3. Systematic Budget-Bounded Local Search
    iterations = 0
    # We use a pointer to cycle through the nodes systematically
    node_idx = 0
    
    while iterations < budget:
        # Focus on one node and its candidate neighbors to find 2-opt swaps
        u = tour[node_idx % n]
        # Get the index of u in the tour
        pos_u = node_idx % n
        
        # The edge starting at u is (u, tour[(pos_u + 1) % n])
        v_next = tour[(pos_u + 1) % n]
        
        # Try swapping (u, v_next) with (u, w) where w is a candidate
        for w in candidates[u]:
            iterations += 1
            if iterations >= budget: break
            
            if w == u or w == v_next:
                continue
                
            # Find position of w in tour
            pos_w = np.where(tour == w)[0][0]
            w_next = tour[(pos_w + 1) % n]
            
            # 2-opt swap: replace (u, v_next) and (w, w_next) with (u, w) and (v_next, w_next)
            # Current distance: dist(u, v_next) + dist(w, w_next)
            # New distance: dist(u, w) + dist(v_next, w_next)
            current_edges = distance_matrix[u, v_next] + distance_matrix[w, w_next]
            new_edges = distance_matrix[u, w] + distance_matrix[v_next, w_next]
            
            if new_edges < current_edges:
                # Perform the 2-opt reverse
                # The segment between v_next and w (inclusive) is reversed
                i, j = pos_u + 1, pos_w
                if i > j:
                    # Wrap around case: reverse the two ends
                    # This is complex, so we simplify by only doing non-wrapping swaps
                    # or use a simpler slice if we normalize the tour to start at 0
                    pass
                else:
                    tour[i : j+1] = tour[i : j+1][::-1]
                    best_dist -= (current_edges - new_edges)
                    report_best_tour(tour)
                    # After an improvement, we can restart the scan or continue
                    # Resetting v_next as the tour changed
                    v_next = tour[(pos_u + 1) % n]

        node_idx += 1

    return tour