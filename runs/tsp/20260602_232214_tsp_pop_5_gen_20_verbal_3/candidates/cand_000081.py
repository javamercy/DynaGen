import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def tour_cost(tour):
        total = distance_matrix[tour[-1], tour[0]]
        for i in range(n-1):
            total += distance_matrix[tour[i], tour[i+1]]
        return total

    rng = np.random.default_rng()
    tour = rng.permutation(n).astype(int)
    current_cost = tour_cost(tour)
    best_tour = tour.copy()
    best_cost = current_cost
    report_best_tour(best_tour)

    T0 = best_cost * 0.1
    if T0 == 0:
        T0 = 1.0
    T = T0
    alpha = 0.9999
    max_iter = n * 10000

    for _ in range(max_iter):
        # 2-opt move: select i < j-1
        i = rng.integers(0, n-2)
        j = rng.integers(i+2, n)
        a = tour[i]
        b = tour[i+1]
        c = tour[j]
        d = tour[(j+1) % n]
        delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Reverse segment (i+1..j)
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = tour.copy()
                report_best_tour(best_tour)
        T *= alpha
        if T < 1e-10:
            break

    return best_tour