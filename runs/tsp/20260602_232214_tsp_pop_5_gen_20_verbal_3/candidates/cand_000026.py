import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    rng = np.random.RandomState()
    tour = rng.permutation(n)
    def tour_cost(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))
    best_tour = tour.copy()
    best_cost = tour_cost(best_tour)
    report_best_tour(best_tour)
    curr_tour = tour.copy()
    curr_cost = best_cost
    T0 = np.max(distance_matrix) * 10
    alpha = 0.999
    max_iter = 100000
    for step in range(max_iter):
        T = T0 * (alpha ** step)
        i, j = rng.randint(n, size=2)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if j == i + 1:
            li = curr_tour[(i-1)%n]
            rj = curr_tour[(j+1)%n]
            a = curr_tour[i]
            b = curr_tour[j]
            old = distance_matrix[li, a] + distance_matrix[a, b] + distance_matrix[b, rj]
            new = distance_matrix[li, b] + distance_matrix[b, a] + distance_matrix[a, rj]
            delta = new - old
        else:
            li = curr_tour[(i-1)%n]
            ri = curr_tour[(i+1)%n]
            lj = curr_tour[(j-1)%n]
            rj = curr_tour[(j+1)%n]
            a = curr_tour[i]
            b = curr_tour[j]
            old = distance_matrix[li, a] + distance_matrix[a, ri] + distance_matrix[lj, b] + distance_matrix[b, rj]
            new = distance_matrix[li, b] + distance_matrix[b, ri] + distance_matrix[lj, a] + distance_matrix[a, rj]
            delta = new - old
        if delta < 0 or rng.random() < np.exp(-delta / T):
            curr_tour[i], curr_tour[j] = curr_tour[j], curr_tour[i]
            curr_cost += delta
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_tour = curr_tour.copy()
                report_best_tour(best_tour)
    return best_tour