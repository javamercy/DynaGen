import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    # nearest neighbor
    def nn_tour():
        start = 0
        tour = [start]
        visited = {start}
        current = start
        for _ in range(n - 1):
            best = None
            best_dist = np.inf
            for v in range(n):
                if v not in visited and distance_matrix[current, v] < best_dist:
                    best_dist = distance_matrix[current, v]
                    best = v
            tour.append(best)
            visited.add(best)
            current = best
        return np.array(tour)

    def cost(tour):
        total = 0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i + 1) % n]]
        return total

    tour = nn_tour()
    current_cost = cost(tour)
    best_tour = tour.copy()
    best_cost = current_cost
    report_best_tour(tour)

    # SA parameters
    T = current_cost * 0.1
    if T == 0:
        T = 1
    alpha = 0.99
    epsilon = 1e-6
    max_iters = n * 20
    rng = np.random.default_rng()

    while T > epsilon:
        improved = False
        for _ in range(max_iters):
            i = rng.integers(0, n - 2)
            j = rng.integers(i + 2, n)
            a, b, c, d = tour[i], tour[i + 1], tour[j], tour[(j + 1) % n]
            delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
            if delta < 0 or rng.random() < np.exp(-delta / T):
                tour = np.concatenate([tour[:i + 1], tour[i + 1:j + 1][::-1], tour[j + 1:]])
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
        T *= alpha

    return best_tour