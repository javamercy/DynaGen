import numpy as np
import random
import heapq

def report_best_tour(tour):
    # Placeholder for the required reporting mechanism
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
        # Find nearest neighbor
        best_next = -1
        min_dist = float('inf')
        # To avoid sorting the whole set, just iterate
        for node in unvisited:
            d = distance_matrix[curr, node]
            if d < min_dist:
                min_dist = d
                best_next = node
        tour.append(best_next)
        unvisited.remove(best_next)
        curr = best_next
    
    tour = np.array(tour)
    report_best_tour(tour)
    
    def get_tour_dist(t):
        d = 0
        for i in range(n):
            d += distance_matrix[t[i], t[(i + 1) % n]]
        return d

    current_dist = get_tour_dist(tour)
    
    # 2. Candidate Lists for scalability
    # Restrict search to k nearest neighbors to keep it efficient
    k_neighbors = 20 if n >= 80 else n - 1
    candidates = []
    for i in range(n):
        # Get indices of k smallest distances from node i
        dists = distance_matrix[i]
        idx = np.argsort(dists)[:k_neighbors + 1]
        candidates.append(set(idx))

    # 3. Budget-bounded Local Search (2-opt)
    iterations = 0
    improved = True
    
    while improved and iterations < budget:
        improved = False
        
        for i in range(n):
            if iterations >= budget: break
            
            u = tour[i]
            v = tour[(i + 1) % n]
            
            # We look for an edge (x, y) to swap with (u, v)
            # To be efficient, we only check edges where x is a neighbor of u
            for x in candidates[u]:
                if x == u or x == v:
                    continue
                
                # Find the position of x in the tour
                # Using np.where is O(n), but the candidate list keeps this loop tight
                pos_x = np.where(tour == x)[0][0]
                y = tour[(pos_x + 1) % n]
                
                if y == u or y == v:
                    continue
                
                # Delta for 2-opt: remove (u,v) and (x,y), add (u,x) and (v,y)
                delta = (distance_matrix[u, x] + distance_matrix[v, y]) - \
                        (distance_matrix[u, v] + distance_matrix[x, y])
                
                if delta < -1e-9:
                    # Perform 2-opt swap: reverse segment from v to x
                    # The segment is from index (i+1) to pos_x
                    idx1, idx2 = (i + 1) % n, pos_x
                    
                    # To handle wrap-around simply, we use a list and slice
                    t_list = tour.tolist()
                    if idx1 < idx2:
                        t_list[idx1 : idx2 + 1] = t_list[idx2 : idx1 - 1 : -1] if idx1 > 0 else t_list[idx2 : : -1]
                        # Correction for slice logic: the standard way is simpler
                        # Let's use a robust manual reverse
                        l, r = idx1, idx2
                        while l < r:
                            t_list[l], t_list[r] = t_list[r], t_list[l]
                            l += 1
                            r -= 1
                    else:
                        # Wrap around: reverse from idx1 to end, and from 0 to idx2
                        # This is equivalent to reversing the middle section in a shifted tour
                        # Simplest way: rotate, reverse, rotate back
                        # But for TSP, just reversing the indices between the two edges works
                        # Let's use the property that reversing [idx1, n-1] and [0, idx2]
                        # is the same as reversing the whole tour then the middle
                        # Alternatively, just use a temporary list and handle indices
                        l, r = idx1, idx2
                        # Since it wraps, we can't use a simple while l < r
                        # We use a temporary array to perform the swap
                        new_tour = np.zeros(n, dtype=int)
                        # The new tour is: tour[0...i], tour[pos_x...i+1 reversed], tour[pos_x+1...n-1]
                        # Actually, let's just use the standard 2-opt reversal on a linear array
                        # by shifting the tour so the swap doesn't wrap
                        shift = i + 1
                        shifted_tour = np.roll(tour, -shift)
                        # Now the edge (u,v) is at (n-1, 0). The edge (x,y) is at (pos_x-shift, pos_x-shift+1)
                        # This is getting complex. Let's use a simple index-based reverse:
                        curr_idx = idx1
                        while curr_idx != idx2:
                            # This is not a simple reverse. Let's stick to the most robust 2-opt:
                            # Reverse the segment between the two edges.
                            break
                    
                    # Re-implementing robust reversal
                    t_list = tour.tolist()
                    l, r = (i + 1) % n, pos_x
                    if l <= r:
                        while l < r:
                            t_list[l], t_list[r] = t_list[r], t_list[l]
                            l += 1
                            r -= 1
                    else:
                        # Wrap around case: reverse from l to n-1 and 0 to r
                        # This is equivalent to reversing the segment [r+1, l-1]
                        # and then reversing the whole tour
                        # Easier: just swap elements manually
                        # The elements to be reversed are: l, l+1...n-1, 0, 1...r
                        seq = []
                        idx = l
                        while idx != (r + 1) % n:
                            seq.append(idx)
                            idx = (idx + 1) % n
                        # This is too slow. Let's use a simpler 2-opt: only swap if not wrapping
                        # or use a fixed reversal logic
                        pass

                    # Simplified robust 2-opt reversal
                    t_list = tour.tolist()
                    a, b = (i + 1) % n, pos_x
                    if a < b:
                        t_list[a:b+1] = t_list[a:b+1][::-1]
                    else:
                        # Wrap around: reverse the part that DOESN'T wrap
                        # and then reverse the whole thing
                        t_list[b+1:a] = t_list[b+1:a][::-1]
                        t_list[:] = t_list[::-1]
                        # Then we must fix the shift to keep node 0 at the start
                        # But for TSP, the tour is a cycle, so we just need to return a valid permutation
                    
                    tour = np.array(t_list)
                    current_dist += delta
                    report_best_tour(tour)
                    improved = True
                    iterations += 1
                    break
            if improved: break
            iterations += 1

    return tour