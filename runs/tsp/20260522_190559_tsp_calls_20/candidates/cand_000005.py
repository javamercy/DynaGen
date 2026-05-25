import numpy as np
import random

def report_best_tour(tour):
    # Placeholder as per requirements
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
        # Use a simple min search
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
    k_neighbors = 25 if n >= 80 else n - 1
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        idx = np.argsort(dists)[:k_neighbors + 1]
        candidates.append(set(idx))

    # 3. Budget-bounded Local Search
    # Budget is treated as the maximum number of successful 2-opt swaps
    # to avoid infinite loops and respect the search limit.
    iterations = 0
    improved = True
    while improved and iterations < budget:
        improved = False
        
        # We iterate through edges (i, i+1)
        for i in range(n):
            if iterations >= budget: break
            u = tour[i]
            v = tour[(i + 1) % n]
            
            # Only check candidate neighbors of u to find a potential swap
            for x in candidates[u]:
                if x == u or x == v: continue
                
                # Find position of x in tour
                # Since n is small enough for np.where or a simple search
                pos_x = np.where(tour == x)[0][0]
                y = tour[(pos_x + 1) % n]
                
                # 2-opt delta: remove (u,v) and (x,y), add (u,x) and (v,y)
                delta = (distance_matrix[u, x] + distance_matrix[v, y]) - \
                        (distance_matrix[u, v] + distance_matrix[x, y])
                
                if delta < -1e-9:
                    # Robust 2-opt reversal: reverse segment from i+1 to pos_x
                    # Indices are relative to the tour array
                    idx1, idx2 = (i + 1) % n, pos_x
                    
                    # To handle wrap-around simply, we convert to list and reverse
                    t_list = tour.tolist()
                    if idx1 <= idx2:
                        t_list[idx1 : idx2 + 1] = t_list[idx2 : idx1 - 1 if idx1 > 0 else None : -1]
                    else:
                        # Wrap-around reversal: reverse from idx1 to end AND 0 to idx2
                        # Simplest way: shift tour so idx1 is at 0, reverse, shift back
                        # Or just use a temporary array
                        new_t = np.zeros(n, dtype=int)
                        # Elements from idx2+1 to idx1-1 remain same relative to each other
                        # The segment [idx1...n-1, 0...idx2] is reversed
                        segment = []
                        for j in range(idx1, n):
                            segment.append(t_list[j])
                        for j in range(0, idx2 + 1):
                            segment.append(t_list[j])
                        
                        reversed_seg = segment[::-1]
                        # Put reversed segment back
                        for j in range(len(reversed_seg)):
                            pos = (idx1 + j) % n
                            t_list[pos] = reversed_seg[j]

                    tour = np.array(t_list)
                    current_dist += delta
                    report_best_tour(tour)
                    improved = True
                    iterations += 1
                    break
            if improved: break
    
    return tour