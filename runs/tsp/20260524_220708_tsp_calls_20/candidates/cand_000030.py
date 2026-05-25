import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)
    # Randomized nearest neighbor construction
    start = rng.integers(n)
    tour = [start]
    unvisited = set(range(n)) - {start}
    while unvisited:
        last = tour[-1]
        min_dist = np.inf
        candidates = []
        for city in unvisited:
            d = distance_matrix[last, city]
            if d < min_dist:
                min_dist = d
                candidates = [city]
            elif d == min_dist:
                candidates.append(city)
        chosen = rng.choice(candidates) if len(candidates) > 1 else candidates[0]
        tour.append(chosen)
        unvisited.remove(chosen)
    tour = np.array(tour, dtype=np.int64)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)
    
    ops = 0
    no_improve_scan = 0
    while ops < budget:
        # First-improvement 2-opt
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
                    cur_dist = 0.0
                    for k in range(n):
                        cur_dist += distance_matrix[tour[k], tour[(k+1)%n]]
                    if cur_dist < best_dist - 1e-12:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
        if not improved:
            no_improve_scan += 1
            if ops >= budget:
                break
            ops += 1  # count perturbation as one operation
            # Random perturbation
            if rng.random() < 0.5:
                # double-bridge
                cuts = sorted(rng.choice(range(1, n), size=4, replace=False))
                a, b, c, d = cuts
                tour = np.concatenate([tour[:a], tour[c:d], tour[b:c], tour[a:b], tour[d:]])
            else:
                # random segment reversal
                i = rng.integers(n)
                j = rng.integers(i+2, n)
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
            # Periodic restart after 5 stagnation scans
            if no_improve_scan >= 5:
                # rebuild tour from randomized nearest neighbor
                start = rng.integers(n)
                tour = [start]
                unvisited = set(range(n)) - {start}
                while unvisited:
                    last = tour[-1]
                    min_dist = np.inf
                    candidates = []
                    for city in unvisited:
                        d = distance_matrix[last, city]
                        if d < min_dist:
                            min_dist = d
                            candidates = [city]
                        elif d == min_dist:
                            candidates.append(city)
                    chosen = rng.choice(candidates) if len(candidates) > 1 else candidates[0]
                    tour.append(chosen)
                    unvisited.remove(chosen)
                tour = np.array(tour, dtype=np.int64)
                no_improve_scan = 0
        else:
            no_improve_scan = 0
    return best_tour