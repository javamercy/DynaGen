import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    # nearest neighbor tour
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda v: distance_matrix[current, v])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    tour = np.array(tour)

    best_tour = tour.copy()
    current_cost = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    best_cost = current_cost
    report_best_tour(best_tour)

    T = best_cost * 0.01
    if T == 0:
        T = 1.0
    alpha = 0.995
    epsilon = 1e-8
    max_iters = n * 100

    rng = np.random.default_rng()
    while T > epsilon:
        for _ in range(max_iters):
            i = rng.integers(0, n - 2)
            j = rng.integers(i + 2, n)
            a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
            delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
            if delta < 0 or rng.random() < np.exp(-delta / T):
                tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
        T *= alpha

    return best_tour