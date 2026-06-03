import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def compute_cost(tour):
        total = 0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i + 1) % n]]
        return total

    def nearest_neighbor(start):
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

    best_tour = None
    best_cost = np.inf
    rng = np.random.default_rng()

    for restart in range(min(3, n)):  # up to 3 restarts
        start_node = rng.integers(n)
        tour = nearest_neighbor(start_node)
        current_cost = compute_cost(tour)
        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = tour.copy()
            report_best_tour(best_tour)

        T = current_cost * 0.001
        if T == 0:
            T = 1.0
        alpha = 0.99
        epsilon = 1e-8
        max_iters = n * 50
        stagnation = 0
        while T > epsilon and stagnation < n:
            improved = False
            for _ in range(max_iters):
                i = rng.integers(0, n - 2)
                j = rng.integers(i + 2, n)
                a, b, c, d = tour[i], tour[i + 1], tour[j], tour[(j + 1) % n]
                delta = (distance_matrix[a, c] + distance_matrix[b, d] -
                         distance_matrix[a, b] - distance_matrix[c, d])
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    tour = np.concatenate([tour[:i + 1], tour[i + 1:j + 1][::-1], tour[j + 1:]])
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                        improved = True
            if improved:
                stagnation = 0
            else:
                stagnation += 1
            T *= alpha

    return best_tour