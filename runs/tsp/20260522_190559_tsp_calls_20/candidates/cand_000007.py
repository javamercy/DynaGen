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

    # 1. Initial Incumbent: Greedy Nearest Neighbor
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
    
    def get_tour_dist(t):
        d = 0
        for i in range(n):
            d += distance_matrix[t[i], t[(i + 1) % n]]
        return d

    current_dist = get_tour_dist(tour)
    
    # Position map for O(1) lookup
    pos_map = np.zeros(n, dtype=int)
    for i in range(n):
        pos_map[tour[i]] = i

    # Candidate lists for pruning
    k_neighbors = 20 if n >= 80 else n - 1
    candidates = []
    for i in range(n):
        idx = np.argsort(distance_matrix[i])[:k_neighbors + 1]
        candidates.append(idx)

    iterations = 0
    improved = True
    
    while improved and iterations < budget:
        improved = False
        
        # Strategy: Alternate between Node Relocation and 2-opt
        # Node Relocation: Move node u to position between v and w
        for u in range(n):
            if iterations >= budget: break
            
            # Current neighbors of u
            p_u = pos_map[u]
            prev_u = tour[(p_u - 1) % n]
            next_u = tour[(p_u + 1) % n]
            
            # Cost of removing u
            removal_cost = distance_matrix[prev_u, u] + distance_matrix[u, next_u] - distance_matrix[prev_u, next_u]
            
            # Try inserting u into candidate edges of its neighbors
            # To keep it efficient, we sample candidate edges
            for v in candidates[u]:
                if v == u: continue
                w = tour[(pos_map[v] + 1) % n]
                if w == u or v == next_u: continue
                
                # Cost of inserting u between v and w
                insertion_cost = distance_matrix[v, u] + distance_matrix[u, w] - distance_matrix[v, w]
                
                delta = insertion_cost - removal_cost
                if delta < -1e-9:
                    # Execute relocation
                    t_list = tour.tolist()
                    t_list.pop(p_u)
                    # Find new insertion index
                    new_pos = (pos_map[v] + 1) % n
                    # Adjust index if pop happened before insertion point
                    if p_u < new_pos:
                        new_pos -= 1
                    if new_pos < 0: new_pos = n - 1
                    
                    t_list.insert(new_pos, u)
                    tour = np.array(t_list)
                    # Update pos_map
                    for i in range(n): pos_map[tour[i]] = i
                    
                    current_dist += delta
                    report_best_tour(tour)
                    improved = True
                    iterations += 1
                    break
            if improved: break

        if improved: continue

        # 2-opt refinement
        for i in range(n):
            if iterations >= budget: break
            u = tour[i]
            v = tour[(i + 1) % n]
            
            for x in candidates[u]:
                if x == u or x == v: continue
                px = pos_map[x]
                y = tour[(px + 1) % n]
                if y == u: continue
                
                delta = (distance_matrix[u, x] + distance_matrix[v, y]) - (distance_matrix[u, v] + distance_matrix[x, y])
                
                if delta < -1e-9:
                    # Reverse segment from i+1 to px
                    idx1, idx2 = (i + 1) % n, px
                    t_list = tour.tolist()
                    if idx1 <= idx2:
                        t_list[idx1 : idx2 + 1] = t_list[idx2 : idx1 - 1 if idx1 > 0 else None : -1]
                    else:
                        segment = t_list[idx1:] + t_list[:idx2 + 1]
                        rev = segment[::-1]
                        for j in range(len(rev)):
                            t_list[(idx1 + j) % n] = rev[j]
                    
                    tour = np.array(t_list)
                    for k in range(n): pos_map[tour[k]] = k
                    current_dist += delta
                    report_best_tour(tour)
                    improved = True
                    iterations += 1
                    break
            if improved: break

    return tour