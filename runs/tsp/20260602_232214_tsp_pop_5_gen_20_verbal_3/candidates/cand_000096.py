import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def cost(tour):
        total = 0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i + 1) % n]]
        return total

    # farthest insertion construction
    start = 0
    tour = [start]
    unvisited = set(range(1, n))
    while unvisited:
        max_min_dist = -1
        best_node = None
        for v in unvisited:
            min_dist = min(distance_matrix[v, t] for t in tour)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_node = v
        best_delta = np.inf
        best_pos = 0
        for i in range(len(tour)):
            nxt = (i + 1) % len(tour)
            delta = distance_matrix[tour[i], best_node] + distance_matrix[best_node, tour[nxt]] - distance_matrix[tour[i], tour[nxt]]
            if delta < best_delta:
                best_delta = delta
                best_pos = i + 1
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
    tour = np.array(tour)
    best_tour = tour.copy()
    best_cost = cost(tour)
    report_best_tour(tour)

    # steepest descent 2-opt
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b, c, d = tour[i], tour[i + 1], tour[j], tour[(j + 1) % n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-12:
                    tour = np.concatenate([tour[:i + 1], tour[i + 1:j + 1][::-1], tour[j + 1:]])
                    best_cost = cost(tour)
                    best_tour = tour.copy()
                    report_best_tour(tour)
                    improved = True
                    break
            if improved:
                break

    # simulated annealing
    current_tour = best_tour.copy()
    current_cost = best_cost
    T = current_cost * 0.01
    if T == 0:
        T = 1.0
    rng = np.random.default_rng()
    alpha = 0.995
    epsilon = 1e-8
    while T > epsilon:
        for _ in range(n * 100):
            i = rng.integers(0, n - 2)
            j = rng.integers(i + 2, n)
            a, b, c, d = current_tour[i], current_tour[i + 1], current_tour[j], current_tour[(j + 1) % n]
            delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_tour = np.concatenate([current_tour[:i + 1], current_tour[i + 1:j + 1][::-1], current_tour[j + 1:]])
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = current_tour.copy()
                    report_best_tour(best_tour)
        T *= alpha
    return best_tour