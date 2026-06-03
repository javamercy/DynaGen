import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour
    # random initial tour
    tour = np.random.permutation(n)
    def tour_distance(t):
        d = 0.0
        for k in range(n):
            d += distance_matrix[t[k], t[(k+1)%n]]
        return d
    current_dist = tour_distance(tour)
    best_dist = current_dist
    best_tour = tour.copy()
    report_best_tour(best_tour)
    # SA parameters
    T0 = 100.0
    T = T0
    cooling_rate = 0.999
    max_iter = 10000
    for _ in range(max_iter):
        i, j = np.random.choice(n, 2, replace=False)
        # swap
        tour[i], tour[j] = tour[j], tour[i]
        new_dist = tour_distance(tour)
        delta = new_dist - current_dist
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_dist = new_dist
            if new_dist < best_dist:
                best_dist = new_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
        else:
            # revert swap
            tour[i], tour[j] = tour[j], tour[i]
        T *= cooling_rate
    return best_tour