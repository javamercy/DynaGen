import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def tour_cost(tour):
        s = 0
        for i in range(n):
            s += distance_matrix[tour[i], tour[(i+1)%n]]
        return s

    # Nearest neighbor from city 0
    tour = [0]
    visited = {0}
    current = 0
    for _ in range(n-1):
        next_city = None
        min_dist = np.inf
        for v in range(n):
            if v not in visited and distance_matrix[current, v] < min_dist:
                min_dist = distance_matrix[current, v]
                next_city = v
        tour.append(next_city)
        visited.add(next_city)
        current = next_city
    tour = np.array(tour)
    cost = tour_cost(tour)
    best_tour = tour.copy()
    best_cost = cost
    report_best_tour(tour)

    rng = np.random.default_rng()

    def improve(tour, cost):
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < 0:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        cost += delta
                        improved = True
                        nonlocal best_cost, best_tour
                        if cost < best_cost:
                            best_cost = cost
                            best_tour[:] = tour[:]
                            report_best_tour(best_tour)
                        break
                if improved:
                    break
        return cost

    cost = improve(tour, cost)

    # Iterated local search: perturb and re-optimize
    for _ in range(5):
        # Perturb with 10 random 2-opt moves
        for _ in range(10):
            i = rng.integers(0, n-2)
            j = rng.integers(i+2, n)
            a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
            delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
            cost += delta
        cost = improve(tour, cost)

    return best_tour