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

    # 1. Construction: Regret-2 Insertion
    start_node = random.randint(0, n - 1)
    unvisited = set(range(n))
    unvisited.remove(start_node)
    
    remaining = list(unvisited)
    second_node = remaining[np.argmin(distance_matrix[start_node, remaining])]
    unvisited.remove(second_node)
    
    tour_list = [start_node, second_node]
    
    while unvisited:
        best_regret = -1
        best_node = -1
        best_pos = -1
        
        candidates = list(unvisited)
        if len(candidates) > 60:
            candidates = random.sample(candidates, 60)

        for node in candidates:
            costs = []
            for i in range(len(tour_list)):
                u = tour_list[i]
                v = tour_list[(i + 1) % len(tour_list)]
                cost = distance_matrix[u, node] + distance_matrix[node, v] - distance_matrix[u, v]
                costs.append(cost)
            
            costs.sort()
            regret = costs[1] - costs[0] if len(costs) > 1 else costs[0]
            
            if regret > best_regret:
                best_regret = regret
                best_node = node
                min_c = float('inf')
                for i in range(len(tour_list)):
                    u = tour_list[i]
                    v = tour_list[(i + 1) % len(tour_list)]
                    c = distance_matrix[u, node] + distance_matrix[node, v] - distance_matrix[u, v]
                    if c < min_c:
                        min_c = c
                        best_pos = i
        
        if best_node == -1:
            best_node = random.choice(list(unvisited))
            best_pos = random.randint(0, len(tour_list) - 1)
            
        tour_list.insert(best_pos + 1, best_node)
        unvisited.remove(best_node)

    current_tour = np.array(tour_list)
    
    def get_tour_dist(t):
        d = 0
        for i in range(n):
            d += distance_matrix[t[i], t[(i+1)%n]]
        return d

    current_dist = get_tour_dist(current_tour)
    best_tour = np.copy(current_tour)
    best_dist = current_dist
    report_best_tour(best_tour)

    # 2. Candidate List Construction
    k_neighbors = 25 if n >= 80 else n - 1
    candidates_list = []
    for i in range(n):
        idx = np.argsort(distance_matrix[i])[:k_neighbors + 1]
        candidates_list.append(idx)

    # 3. Systematic Local Search (Budget-bounded 2-opt)
    iters = 0
    node_ptr = 0
    
    while iters < budget:
        # Systematic node selection
        u_idx = node_ptr % n
        u = current_tour[u_idx]
        v_next = current_tour[(u_idx + 1) % n]
        
        # Try swapping edge (u, v_next) with (u, w) where w is a neighbor
        for w in candidates_list[u]:
            iters += 1
            if iters >= budget: break
            
            if w == u or w == v_next:
                continue
                
            # Find position of w in tour
            # We use np.where for simplicity, though a map would be faster
            pos_w = np.where(current_tour == w)[0][0]
            w_next = current_tour[(pos_w + 1) % n]
            
            # 2-opt delta calculation
            # Replace (u, v_next) and (w, w_next) with (u, w) and (v_next, w_next)
            d_old = distance_matrix[u, v_next] + distance_matrix[w, w_next]
            d_new = distance_matrix[u, w] + distance_matrix[v_next, w_next]
            delta = d_new - d_old
            
            if delta < -1e-9:
                # Perform 2-opt reversal
                # The range to reverse is between the node after u and node w
                i, j = u_idx + 1, pos_w
                if i < j:
                    current_tour[i : j+1] = current_tour[i : j+1][::-1]
                elif i > j:
                    # Wrap-around reversal: reverse the two ends
                    # To simplify, we rotate tour so that u is at 0
                    current_tour = np.roll(current_tour, -u_idx)
                    # Now u is at 0, v_next is at 1. w is at (pos_w - u_idx) % n
                    new_pos_w = (pos_w - u_idx) % n
                    current_tour[1 : new_pos_w + 1] = current_tour[1 : new_pos_w + 1][::-1]
                    # Roll back
                    current_tour = np.roll(current_tour, u_idx)
                
                current_dist += delta
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_tour = np.copy(current_tour)
                    report_best_tour(best_tour)
                
                # Update v_next for the current node u
                v_next = current_tour[(u_idx + 1) % n]

        node_ptr += 1

    return best_tour