import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour

    def tour_dist(tour):
        d = 0.0
        for k in range(n-1):
            d += distance_matrix[tour[k], tour[k+1]]
        d += distance_matrix[tour[-1], tour[0]]
        return d

    def nn_tour(start):
        tour = np.full(n, -1, dtype=np.int32)
        unvisited = np.ones(n, dtype=bool)
        tour[0] = start
        unvisited[start] = False
        current = start
        for i in range(1, n):
            dists = np.where(unvisited, distance_matrix[current], np.inf)
            next_node = np.argmin(dists)
            tour[i] = next_node
            unvisited[next_node] = False
            current = next_node
        return tour

    def two_opt(tour):
        ext = np.concatenate([tour, [tour[0]]])
        improved = True
        passes = 0
        max_passes = 2 * n
        while improved and passes < max_passes:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    delta = (distance_matrix[ext[i], ext[j]] +
                             distance_matrix[ext[i+1], ext[j+1]] -
                             distance_matrix[ext[i], ext[i+1]] -
                             distance_matrix[ext[j], ext[j+1]])
                    if delta < -1e-12:
                        ext[i+1:j+1] = ext[i+1:j+1][::-1]
                        improved = True
            passes += 1
        return ext[:n].copy()

    # Multi-start NN + 2-opt
    starts = [0]
    if n >= 4:
        starts.extend([n//4, n//2, 3*n//4])
    else:
        starts = list(range(n))

    best_tour = None
    best_dist = np.inf

    for start in starts:
        tour = nn_tour(start)
        tour = two_opt(tour)
        d = tour_dist(tour)
        if d < best_dist - 1e-12:
            best_dist = d
            best_tour = tour.copy()
            report_best_tour(best_tour)

    # Iterated local search with random perturbation
    if n >= 3:
        rng = np.random.default_rng(42)
        max_iter = max(10, n // 10)
        for _ in range(max_iter):
            i = rng.integers(0, n-2)
            j = rng.integers(i+2, n)
            perturbed = best_tour.copy()
            perturbed[i:j+1] = perturbed[i:j+1][::-1]
            perturbed = two_opt(perturbed)
            d = tour_dist(perturbed)
            if d < best_dist - 1e-12:
                best_dist = d
                best_tour = perturbed.copy()
                report_best_tour(best_tour)

    return best_tour