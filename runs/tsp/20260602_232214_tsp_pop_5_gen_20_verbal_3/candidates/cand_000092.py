import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def nn_tour():
        tour = [0]
        unvisited = set(range(1, n))
        last = 0
        while unvisited:
            nearest = min(unvisited, key=lambda x: distance_matrix[last, x])
            tour.append(nearest)
            unvisited.remove(nearest)
            last = nearest
        return np.array(tour)

    def cost(tour):
        total = 0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i+1)%n]]
        return total

    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
                        improved = True
        return tour

    def double_bridge(tour):
        rng = np.random.default_rng()
        a = rng.integers(0, n // 3)
        b = rng.integers(a+1, a + n // 3)
        c = rng.integers(b+1, b + n // 3)
        d = rng.integers(c+1, c + n // 3)
        segments = [tour[:a], tour[a:b], tour[b:c], tour[c:d], tour[d:]]
        tour = np.concatenate([segments[0], segments[3], segments[2], segments[1], segments[4]])
        return tour

    tour = nn_tour()
    best_tour = tour.copy()
    best_cost = cost(tour)
    report_best_tour(best_tour)

    for _ in range(10):
        tour = two_opt(tour)
        current_cost = cost(tour)
        if current_cost < best_cost - 1e-12:
            best_cost = current_cost
            best_tour = tour.copy()
            report_best_tour(best_tour)
        tour = double_bridge(best_tour)

    return best_tour