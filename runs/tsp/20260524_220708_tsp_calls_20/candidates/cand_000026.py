import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        np.random.default_rng(seed).shuffle(tour)
        return tour
    rng = np.random.default_rng(seed)
    # nearest neighbor construction
    start = rng.integers(n)
    tour = [start]
    unvisited = set(range(n)) - {start}
    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda c: distance_matrix[current, c])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    tour = np.array(tour, dtype=np.int64)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)
    ops = 0
    improved = True
    stagnation = 0
    threshold = n  # fixed stagnation limit
    while ops < budget:
        if not improved:
            stagnation += 1
            if stagnation > threshold:
                # double-bridge perturbation
                indices = rng.choice(n-1, size=4, replace=False) + 1
                i, j, k, l = sorted(indices)
                seg1 = tour[i:j]
                seg2 = tour[j:k]
                seg3 = tour[k:l]
                seg4 = np.concatenate([tour[l:], tour[:i]])
                tour = np.concatenate([seg1, seg3, seg2, seg4])
                cur_dist = 0.0
                for t in range(n):
                    cur_dist += distance_matrix[tour[t], tour[(t+1)%n]]
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                improved = True
                stagnation = 0
                ops += 1
                continue
        else:
            stagnation = 0
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