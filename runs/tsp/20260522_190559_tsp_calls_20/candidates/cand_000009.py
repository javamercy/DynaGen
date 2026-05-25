import numpy as np
import random

def report_best_tour(tour):
    # Placeholder for reporting mechanism
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
    tour_list = [0]
    curr = 0
    while unvisited:
        # Find nearest neighbor efficiently
        next_node = min(unvisited, key=lambda x: distance_matrix[curr, x])
        tour_list.append(next_node)
        unvisited.remove(next_node)
        curr = next_node
    
    tour = np.array(tour_list)
    report_best_tour(tour)

    def get_tour_dist(t):
        d = 0
        for i in range(n):
            d += distance_matrix[t[i], t[(i + 1) % n]]
        return d

    current_dist = get_tour_dist(tour)

    # 2. Candidate Lists for scalability
    # Only consider k nearest neighbors for each node to prune search space
    k_neighbors = 25 if n >= 80 else n - 1
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        idx = np.argsort(dists)[:k_neighbors + 1]
        candidates.append(set(idx))

    # 3. Budget-bounded Local Search (2-opt)
    # Maintain a position map for O(1) node lookup
    pos_map = np.zeros(n, dtype=int)
    for i in range(n):
        pos_map[tour[i]] = i

    iterations = 0
    improved = True
    while improved and iterations < budget:
        improved = False
        
        # Iterate through edges (i, i+1)
        for i in range(n):
            if iterations >= budget: break
            u = tour[i]
            v = tour[(i + 1) % n]
            
            # Check candidate neighbors of u to find a potential swap
            for x in candidates[u]:
                if x == u or x == v:
                    continue
                
                pos_x = pos_map[x]
                y = tour[(pos_x + 1) % n]
                
                if y == u or y == v:
                    continue
                
                # Delta for 2-opt: remove (u,v) and (x,y), add (u,x) and (v,y)
                delta = (distance_matrix[u, x] + distance_matrix[v, y]) - \
                        (distance_matrix[u, v] + distance_matrix[x, y])
                
                if delta < -1e-9:
                    # Robust 2-opt reversal: reverse segment from i+1 to pos_x
                    idx1, idx2 = (i + 1) % n, pos_x
                    
                    t_list = tour.tolist()
                    if idx1 <= idx2:
                        t_list[idx1 : idx2 + 1] = t_list[idx2 : idx1 - 1 if idx1 > 0 else None : -1]
                    else:
                        # Wrap-around reversal: reverse from idx1 to end AND 0 to idx2
                        segment = t_list[idx1:] + t_list[:idx2 + 1]
                        reversed_seg = segment[::-1]
                        for j in range(len(reversed_seg)):
                            t_list[(idx1 + j) % n] = reversed_seg[j]
                    
                    tour = np.array(t_list)
                    # Update position map
                    for idx in range(n):
                        pos_map[tour[idx]] = idx
                    
                    current_dist += delta
                    report_best_tour(tour)
                    improved = True
                    iterations += 1
                    break
            if improved: break

    return tour