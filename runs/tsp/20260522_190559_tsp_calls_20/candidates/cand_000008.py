import numpy as np
import random

def report_best_tour(tour):
    # Required reporting mechanism
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
    k_neighbors = 20 if n >= 80 else n - 1
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        idx = np.argsort(dists)[:k_neighbors + 1]
        candidates.append(list(idx))

    # Position map for O(1) lookup
    pos_map = np.zeros(n, dtype=int)
    for i in range(n):
        pos_map[tour[i]] = i

    # 3. Budget-bounded Local Search
    iterations = 0
    improved = True
    while improved and iterations < budget:
        improved = False
        
        # Randomize search order of nodes to diversify search
        node_order = list(range(n))
        random.shuffle(node_order)
        
        for u in node_order:
            if iterations >= budget: break
            
            pos_u = pos_map[u]
            v = tour[(pos_u + 1) % n]
            
            # Only check candidates for u to find a better swap
            # Shuffle candidates to avoid deterministic local minima
            cand_u = candidates[u]
            random.shuffle(cand_u)
            
            for w in cand_u:
                if iterations >= budget: break
                if w == u or w == v: continue
                
                pos_w = pos_map[w]
                x = tour[(pos_w + 1) % n]
                
                # Delta for 2-opt: remove (u,v) and (w,x), add (u,w) and (v,x)
                delta = (distance_matrix[u, w] + distance_matrix[v, x]) - \
                        (distance_matrix[u, v] + distance_matrix[w, x])
                
                if delta < -1e-9:
                    # Perform 2-opt swap: reverse segment between v and w
                    # Indices of the segment to reverse are (pos_u+1) and pos_w
                    idx1 = (pos_u + 1) % n
                    idx2 = pos_w
                    
                    # Handle reversal carefully
                    if idx1 <= idx2:
                        # Simple case: segment is contiguous
                        tour[idx1:idx2+1] = tour[idx2:idx1-1:-1] if idx1 > 0 else tour[idx2::-1]
                    else:
                        # Wrap-around case: reverse the ends and the middle
                        # Easier to just use a list and slice it
                        t_list = tour.tolist()
                        # Reverse from idx1 to end, and start to idx2
                        # But the standard 2-opt logic is simply reversing the path
                        # between the two edges. In a circular array, we can rotate
                        # so that the break is not in the middle of the reversed segment.
                        # For simplicity, we rotate the tour
                        rotated = np.roll(tour, -idx1)
                        rev_len = (idx2 - idx1 + n) % n + 1
                        rotated[:rev_len] = rotated[:rev_len][::-1]
                        tour = np.roll(rotated, idx1)

                    # Update position map
                    for i in range(n):
                        pos_map[tour[i]] = i
                        
                    current_dist += delta
                    report_best_tour(tour)
                    improved = True
                    iterations += 1
                    break
                iterations += 1
            if improved: break

    return tour