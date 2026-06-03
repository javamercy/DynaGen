import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    rng = np.random.default_rng()
    best_tour = None
    best_dist = float('inf')
    restarts = max(5, n // 10)
    for _ in range(restarts):
        start = rng.integers(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            dists = distance_matrix[current, list(unvisited)]
            nearest_idx = list(unvisited)[np.argmin(dists)]
            tour.append(nearest_idx)
            unvisited.remove(nearest_idx)
            current = nearest_idx
        tour = np.array(tour)
        def two_opt(tour):
            improved = True
            while improved:
                improved = False
                for i in range(n-2):
                    for j in range(i+2, n):
                        a, b = tour[i], tour[i+1]
                        c, d = tour[j], tour[(j+1) % n]
                        delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                        if delta < -1e-10:
                            tour[i+1:j+1] = tour[j:i:-1]
                            improved = True
                            report_best_tour(tour)
            return tour
        tour = two_opt(tour)
        dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour