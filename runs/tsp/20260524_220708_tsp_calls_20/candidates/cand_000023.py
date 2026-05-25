import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)
    # Regret-based construction
    start = rng.integers(n)
    tour_list = [start]
    unvisited = set(range(n)) - {start}
    while unvisited:
        best_insert = {}
        for city in unvisited:
            best_cost = np.inf
            second_best = np.inf
            best_pos = -1
            m = len(tour_list)
            for i in range(m):
                prev = tour_list[i]
                nxt = tour_list[(i+1) % m]
                inc = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                if inc < best_cost:
                    second_best = best_cost
                    best_cost = inc
                    best_pos = i+1
                elif inc < second_best:
                    second_best = inc
            regret = second_best - best_cost if second_best != np.inf else best_cost
            best_insert[city] = (regret, best_pos, best_cost)
        chosen = max(unvisited, key=lambda c: best_insert[c][0])
        pos = best_insert[chosen][1]
        tour_list.insert(pos, chosen)
        unvisited.remove(chosen)
    tour = np.array(tour_list, dtype=np.int64)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)
    
    def evaluate_dist(t):
        d = 0.0
        for i in range(n):
            d += distance_matrix[t[i], t[(i+1)%n]]
        return d
    
    ops = 0
    improved = True
    while ops < budget:
        if not improved:
            if ops >= budget:
                break
            # Double-bridge perturbation
            a = rng.integers(0, n-3)
            b = rng.integers(a+1, n-2)
            c = rng.integers(b+1, n-1)
            d = rng.integers(c+1, n)
            # Build new tour: segments [0..a], [c+1..d], [b+1..c], [a+1..b], [d+1..n-1]
            # But careful: indices with wrap-around? We treat array as linear.
            # Use concatenation of slices (as lists) for simplicity.
            t = tour.tolist()
            seg1 = t[:a+1]
            seg2 = t[c+1:d+1]
            seg3 = t[b+1:c+1]
            seg4 = t[a+1:b+1]
            seg5 = t[d+1:]
            new_tour = np.array(seg1 + seg2 + seg3 + seg4 + seg5, dtype=np.int64)
            tour = new_tour
            ops += 1  # count perturbation as an operation
            improved = True
            continue
        improved = False
        for i in range(n-1):
            if ops >= budget:
                break
            for j in range(i+2, n):
                if ops >= budget:
                    break
                ops += 1
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                if new < old - 1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    cur_dist = best_dist + (new - old)
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour