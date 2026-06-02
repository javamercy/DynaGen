import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)
    
    best_tour = None
    best_cost = np.inf
    rcl_size = max(2, int(np.sqrt(n)))
    max_restarts = 10
    
    for _ in range(max_restarts):
        # Randomized greedy construction
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            dists = distance_matrix[current, list(unvisited)]
            sorted_indices = np.argsort(dists)
            sorted_cities = np.array(list(unvisited))[sorted_indices]
            k = min(rcl_size, len(sorted_cities))
            candidate = sorted_cities[np.random.randint(k)]
            tour.append(candidate)
            unvisited.remove(candidate)
            current = candidate
        tour = np.array(tour, dtype=np.int64)
        
        # Helper delta functions
        def two_opt_delta(i, j):
            # assumes i < j, not adjacent
            a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
            old = distance_matrix[a, b] + distance_matrix[c, d]
            new = distance_matrix[a, c] + distance_matrix[b, d]
            return new - old
        
        def apply_two_opt(i, j):
            tour[i+1:j+1] = np.flip(tour[i+1:j+1])
        
        def or_opt_delta(pos, seg_start, seg_end):
            # move segment [seg_start, seg_end] after pos
            L = seg_end - seg_start + 1
            if pos < seg_start:
                # segment is after pos
                a = tour[pos]
                # last of segment
                x = tour[seg_end]
                # first after segment
                y = tour[(seg_end+1)%n]
                old = distance_matrix[a, tour[(pos+1)%n]] + distance_matrix[x, y]
                new = distance_matrix[a, tour[seg_start]] + distance_matrix[tour[seg_end], tour[(pos+1)%n]]
            else:
                # segment is before pos
                a = tour[pos]
                b = tour[(pos+1)%n]
                # first of segment
                x = tour[seg_start]
                # before segment
                y = tour[(seg_start-1+n)%n]
                old = distance_matrix[y, x] + distance_matrix[a, b]
                new = distance_matrix[y, a] + distance_matrix[tour[seg_end], b]
            return new - old
        
        def apply_or_opt(pos, seg_start, seg_end):
            seg = tour[seg_start:seg_end+1].copy()
            if pos < seg_start:
                # remove seg, insert after pos
                tour = np.concatenate([tour[:pos+1], seg, tour[pos+1:seg_start], tour[seg_end+1:]])
            else:
                # remove seg, insert after pos (which is after seg in original)
                tour = np.concatenate([tour[:seg_start], tour[seg_end+1:pos+1], seg, tour[pos+1:]])
            return tour
        
        # Local search: alternating 2-opt and Or-opt until no improvement
        improved = True
        while improved:
            improved = False
            # 2-opt first improvement
            for i in range(n-1):
                for j in range(i+2, n):
                    if j == i+1:
                        continue
                    delta = two_opt_delta(i, j)
                    if delta < 0:
                        apply_two_opt(i, j)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Or-opt first improvement (segment lengths 1,2,3)
            for L in [1,2,3]:
                for seg_start in range(n):
                    seg_end = (seg_start + L - 1) % n
                    if seg_end < seg_start:  # wrap-around, skip for simplicity
                        continue
                    for pos in range(n):
                        # skip overlapping positions
                        if (pos >= seg_start and pos <= seg_end) or (pos == (seg_end+1)%n) or (pos == (seg_start-1+n)%n):
                            continue
                        delta = or_opt_delta(pos, seg_start, seg_end)
                        if delta < 0:
                            tour = apply_or_opt(pos, seg_start, seg_end)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            # Note: after modifying tour, we need to recompute n? n unchanged
        # Compute cost
        cost = distance_matrix[tour[-1], tour[0]]
        for k in range(n-1):
            cost += distance_matrix[tour[k], tour[k+1]]
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(best_tour)
    
    return best_tour