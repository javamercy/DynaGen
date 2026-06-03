import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def cheapest_insertion_tour():
        start = 0
        tour = [start]
        unvisited = set(range(1, n))
        while unvisited:
            best_delta = np.inf
            best_node = None
            best_pos = None
            for v in unvisited:
                for i in range(len(tour)):
                    next_i = (i + 1) % len(tour)
                    delta = distance_matrix[tour[i], v] + distance_matrix[v, tour[next_i]] - distance_matrix[tour[i], tour[next_i]]
                    if delta < best_delta:
                        best_delta = delta
                        best_node = v
                        best_pos = i + 1
            tour.insert(best_pos, best_node)
            unvisited.remove(best_node)
        return np.array(tour)

    def cost(tour):
        total = 0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i + 1) % n]]
        return total

    tour = cheapest_insertion_tour()
    current_cost = cost(tour)
    best_tour = tour.copy()
    best_cost = current_cost
    report_best_tour(tour)

    T = current_cost * 0.1
    if T == 0:
        T = 1.0
    rng = np.random.default_rng()
    alpha = 0.999
    epsilon = 1e-10
    max_iters = n * 50
    while T > epsilon:
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