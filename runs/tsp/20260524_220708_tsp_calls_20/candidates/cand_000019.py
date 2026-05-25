import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(tour)
        report_best_tour(tour)
        return tour
    rng = np.random.default_rng(seed)
    best_tour = None
    best_dist = np.inf
    ops = 0
    first = True
    while ops < budget:
        if first:
            start = rng.integers(n)
            tour = [start]
            unvisited = set(range(n)) - {start}
            while unvisited:
                last = tour[-1]
                next_city = min(unvisited, key=lambda c: distance_matrix[last, c])
                tour.append(next_city)
                unvisited.remove(next_city)
            tour = np.array(tour, dtype=np.int64)
            first = False
        else:
            tour = rng.permutation(n).astype(np.int64)
        cur_dist = 0.0
        for i in range(n):
            cur_dist += distance_matrix[tour[i], tour[(i+1)%n]]
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        improved = True
        while improved and ops < budget:
            improved = False
            for i in range(n-1):
                if ops >= budget:
                    break
                for j in range(i+2, n):
                    if ops >= budget:
                        break
                    ops += 1
                    a, b = tour[i], tour[i+1]
                    c, d = tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        cur_dist += delta
                        if cur_dist < best_dist - 1e-12:
                            best_dist = cur_dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
    return best_tour