import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def tour_cost(tour):
        cost = 0
        for i in range(n):
            cost += distance_matrix[tour[i], tour[(i+1)%n]]
        return cost

    def nearest_neighbor():
        start = 0
        tour = [start]
        visited = {start}
        cur = start
        for _ in range(n-1):
            best_node = None
            best_dist = np.inf
            for v in range(n):
                if v not in visited and distance_matrix[cur, v] < best_dist:
                    best_dist = distance_matrix[cur, v]
                    best_node = v
            tour.append(best_node)
            visited.add(best_node)
            cur = best_node
        return np.array(tour)

    def two_opt(tour):
        improved = True
        cost = tour_cost(tour)
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-12:
                        tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
                        cost += delta
                        improved = True
        return tour, cost

    best_tour = None
    best_cost = np.inf
    rng = np.random.default_rng()

    initial_tour = nearest_neighbor()
    tour, cost = two_opt(initial_tour)
    if cost < best_cost:
        best_cost = cost
        best_tour = tour.copy()
        report_best_tour(best_tour)

    for _ in range(4):
        i = rng.integers(0, n-1)
        j = rng.integers(i+2, n)
        perturbed = np.concatenate([best_tour[:i+1], best_tour[i+1:j+1][::-1], best_tour[j+1:]])
        tour, cost = two_opt(perturbed)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(best_tour)

    return best_tour