import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    rng = np.random.default_rng()
    tour = rng.permutation(n)
    def tour_cost(t):
        total = 0
        for i in range(n):
            total += distance_matrix[t[i], t[(i+1)%n]]
        return total
    current_cost = tour_cost(tour)
    best_tour = tour.copy()
    best_cost = current_cost
    report_best_tour(tour)
    T0 = current_cost * 0.1
    if T0 == 0:
        T0 = 1
    T = T0
    alpha = 0.99
    epsilon = 1e-3
    max_iters_per_temp = n * 50
    while T > epsilon:
        for _ in range(max_iters_per_temp):
            i = rng.integers(0, n-2)
            j = rng.integers(i+2, n)
            a = tour[i]
            b = tour[i+1]
            c = tour[j]
            d = tour[(j+1)%n]
            delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
            if delta < T:
                tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
        T *= alpha
    return best_tour