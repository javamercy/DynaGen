import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # initial random tour
    tour = np.random.permutation(n)
    best_tour = tour.copy()
    best_cost = _tour_cost(distance_matrix, tour)
    report_best_tour(tour)
    # simulated annealing parameters
    T = 1.0
    T_min = 1e-6
    alpha = 0.99
    max_iter = 10000
    for _ in range(max_iter):
        if T < T_min:
            break
        # generate neighbor by swapping two random cities
        i, j = np.random.choice(n, 2, replace=False)
        new_tour = tour.copy()
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_cost = _tour_cost(distance_matrix, new_tour)
        delta = new_cost - _tour_cost(distance_matrix, tour)  # actually compute based on current tour? Better compute directly
        # but we already have new_cost and current cost, so delta = new_cost - current_cost
        current_cost = _tour_cost(distance_matrix, tour)
        delta = new_cost - current_cost
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            tour = new_tour
            if new_cost < best_cost:
                best_cost = new_cost
                best_tour = tour.copy()
                report_best_tour(best_tour)
        T *= alpha
    return best_tour

def _tour_cost(dm, tour):
    n = len(tour)
    cost = dm[tour[-1], tour[0]]
    for k in range(n-1):
        cost += dm[tour[k], tour[k+1]]
    return cost