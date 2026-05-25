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
    k_neighbors = 25 if n >= 80 else n - 1
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        idx = np.argsort(dists)[:k_neighbors + 1]
        candidates.append(set(idx))

    # 3. Budget-bounded Local Search (2-opt)
    pos_map = np.zeros(n, dtype=int)
    for i in range(n):
        pos_map[tour[i]] = i

    iterations = 0
    improved = True
    while improved and iterations < budget:
        improved = False
        
        for i in range(n):
            if iterations >= budget: break
            u = tour[i]
            v = tour[(i + 1) % n]
            
            for x in candidates[u]:
                if x == u or x == v:
                    continue
                
                pos_x = pos_map[x]
                y = tour[(pos_x + 1) % n]
                
                if y == u or y == v:
                    continue
                
                delta = (distance_matrix[u, x] + distance_matrix[v, y]) - \
                        (distance_matrix[u, v] + distance_matrix[x, y])
                
                if delta < -1e-9:
                    # 2-opt reversal
                    idx1, idx2 = (i + 1) % n, pos_x
                    
                    # Extract the segment to reverse
                    if idx1 <= idx2:
                        # Standard segment reversal
                        segment = tour[idx1 : idx2 + 1]
                        reversed_seg = segment[::-1]
                        tour[idx1 : idx2 + 1] = reversed_seg
                        # Efficient pos_map update: only for the reversed nodes
                        for k in range(len(segment)):
                            pos_map[reversed_seg[k]] = idx1 + k
                    else:
                        # Wrap-around reversal
                        # Segment consists of [idx1...n-1] and [0...idx2]
                        seg1 = tour[idx1:]
                        seg2 = tour[:idx2 + 1]
                        combined = np.concatenate([seg1, seg2])
                        reversed_combined = combined[::-1]
                        
                        # Place back into tour
                        split_point = len(seg1)
                        tour[idx1:] = reversed_combined[:split_point]
                        tour[:idx2 + 1] = reversed_combined[split_point:]
                        
                        # Update pos_map for the wrap-around segment
                        for k in range(len(reversed_combined)):
                            actual_idx = (idx1 + k) % n
                            pos_map[reversed_combined[k]] = actual_idx
                    
                    current_dist += delta
                    report_best_tour(tour)
                    improved = True
                    iterations += 1
                    break
            if improved: break

    return tour