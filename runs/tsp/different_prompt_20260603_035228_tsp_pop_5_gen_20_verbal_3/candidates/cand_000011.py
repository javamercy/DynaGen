import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=int)
        report_best_tour(tour.copy())
        return tour
    if n == 2:
        tour = np.array([0, 1], dtype=int)
        report_best_tour(tour.copy())
        return tour

    # Random initial tour
    tour = np.random.permutation(n)
    # Compute initial cost
    cost = distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])
    best_cost = cost
    best_tour = tour.copy()
    report_best_tour(best_tour.copy())

    # Simulated annealing parameters
    T0 = 1.0
    T_end = 1e-3
    alpha = 0.99
    max_iter_per_temp = 100 * n

    T = T0
    while T > T_end:
        for _ in range(max_iter_per_temp):
            i = np.random.randint(0, n - 1)
            j = np.random.randint(i + 2, n)  # j <= n-1, ensures at least 2 edges reversed
            # delta = new cost - old cost
            if j == n - 1:
                delta = (distance_matrix[tour[i], tour[i+1]] + distance_matrix[tour[j], tour[0]]) - (
                         distance_matrix[tour[i], tour[j]] + distance_matrix[tour[i+1], tour[0]])
            else:
                delta = (distance_matrix[tour[i], tour[i+1]] + distance_matrix[tour[j], tour[j+1]]) - (
                         distance_matrix[tour[i], tour[j]] + distance_matrix[tour[i+1], tour[j+1]])
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # Accept move
                new_tour = tour.copy()
                new_tour[i+1:j+1] = new_tour[i+1:j+1][::-1]
                tour = new_tour
                cost -= delta
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour.copy()
                    report_best_tour(best_tour.copy())
        T *= alpha

    return best_tour