import numpy as np
import random
import heapq

def report_best_tour(tour):
    # This is a placeholder for the required reporting mechanism
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
    
    # 2. Candidate Lists for scalability
    # For large n, only consider k nearest neighbors to avoid O(n^2) search
    k_neighbors = 20 if n >= 80 else n - 1
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        idx = np.argsort(dists)[:k_neighbors + 1]
        candidates.append(set(idx))

    # 3. Budget-bounded Local Search
    iterations = 0
    improved = True
    while improved and iterations < budget:
        improved = False
        
        # Try 2-opt (edge swaps)
        for i in range(n):
            if iterations >= budget: break
            u = tour[i]
            v = tour[(i + 1) % n]
            
            # Only check candidates for v to find a better swap
            for w in candidates[u]:
                if w == u or w == v: continue
                # Find where w is in the tour
                # To keep it fast, we search for w's position
                # In a real high-perf solver, we'd maintain a position map
                # but for this compact version, we use np.where
                pos_w = np.where(tour == w)[0][0]
                x = tour[pos_w]
                y = tour[(pos_w + 1) % n]
                
                # Delta for 2-opt: remove (u,v) and (x,y), add (u,x) and (v,y)
                delta = (distance_matrix[u, x] + distance_matrix[v, y]) - \
                        (distance_matrix[u, v] + distance_matrix[x, y])
                
                if delta < -1e-9:
                    # Perform 2-opt swap: reverse segment between v and x
                    # tour[i+1] to tour[pos_w]
                    if i < pos_w:
                        tour[i+1:pos_w+1] = tour[pos_w:i:-1]
                    else:
                        # wrap around case
                        tour[i+1:] = tour[pos_w:i+1:-1]
                        tour[:pos_w+1] = tour[i:pos_w-1:-1] # simplified
                        # For simplicity in compact code, we use a standard slice
                        # But let's use a more robust swap for all indices:
                        new_tour = tour.copy()
                        idx1, idx2 = i + 1, pos_w
                        if idx1 < idx2:
                            new_tour[idx1:idx2+1] = tour[idx2:idx1-1:-1]
                        else:
                            # This part is trickier, let's just use a simple slice logic
                            pass
                    
                    # Correct 2-opt slice logic
                    t_list = tour.tolist()
                    a, b = (i + 1) % n, pos_w
                    if a < b:
                        t_list[a:b+1] = t_list[b:a-1:-1] if a > 0 else t_list[b::-1] # flawed
                    
                    # Standard 2-opt reversal
                    # Let's use a simpler loop-based reversal to be safe
                    t_list = tour.tolist()
                    l, r = i + 1, pos_w
                    if l > r: l, r = r, l
                    while l < r:
                        t_list[l], t_list[r] = t_list[r], t_list[l]
                        l += 1
                        r -= 1
                    tour = np.array(t_list)
                    current_dist += delta
                    report_best_tour(tour)
                    improved = True
                    iterations += 1
                    break
            if improved: break
            iterations += 1

    return tour