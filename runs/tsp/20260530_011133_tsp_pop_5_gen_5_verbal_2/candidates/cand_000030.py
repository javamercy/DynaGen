import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=int)

    def total_distance(tour):
        d = 0.0
        for i in range(n - 1):
            d += distance_matrix[tour[i], tour[i + 1]]
        d += distance_matrix[tour[-1], tour[0]]
        return d

    # Nearest neighbor construction
    tour = np.zeros(n, dtype=int)
    visited = np.zeros(n, bool)
    tour[0] = 0
    visited[0] = True
    for i in range(1, n):
        last = tour[i-1]
        best = -1
        best_dist = np.inf
        for j in range(n):
            if not visited[j] and distance_matrix[last, j] < best_dist:
                best_dist = distance_matrix[last, j]
                best = j
        tour[i] = best
        visited[best] = True

    best_tour = tour.copy()
    best_dist = total_distance(best_tour)
    report_best_tour(best_tour)

    # 2-opt improvement with delta evaluation
    def two_opt(tour):
        improved = True
        max_passes = 100
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    a = tour[i-1]
                    b = tour[i]
                    c = tour[j]
                    d = tour[(j+1) % n]
                    delta = distance_matrix[a, b] + distance_matrix[c, d] - distance_matrix[a, c] - distance_matrix[b, d]
                    if delta > 1e-12:
                        tour = np.concatenate([tour[:i], tour[i:j+1][::-1], tour[j+1:]])
                        improved = True
                        # report if better
                        if delta > 0:
                            pass
                        break  # restart scanning after change
            if improved:
                pass
        return tour

    best_tour = two_opt(best_tour.copy())
    best_dist = total_distance(best_tour)
    report_best_tour(best_tour)

    # Perturbation and local search
    for _ in range(10):
        # double-bridge perturbation
        p = np.random.randint(1, n // 2)
        q = np.random.randint(p + 1, n - 2)
        r = np.random.randint(q + 1, n - 1)
        if p < 1 or q <= p or r <= q:
            continue
        new_tour = np.concatenate([best_tour[:p], best_tour[r:], best_tour[q:r], best_tour[p:q]])
        new_tour = two_opt(new_tour)
        new_dist = total_distance(new_tour)
        if new_dist < best_dist - 1e-12:
            best_tour = new_tour
            best_dist = new_dist
            report_best_tour(best_tour)

    return best_tour