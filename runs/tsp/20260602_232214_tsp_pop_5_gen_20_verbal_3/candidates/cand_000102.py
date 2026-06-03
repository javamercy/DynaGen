import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    # Cheapest insertion
    tour = [0, 1]
    cost = distance_matrix[0,1] + distance_matrix[1,0]  # actually just 2*dist? correct: total distance is sum of edges, but we only need symmetric? anyway, insertion cost delta will be used
    unvisited = set(range(2, n))
    while unvisited:
        best_delta = np.inf
        best_city = None
        best_pos = None
        for v in unvisited:
            for i in range(len(tour)):
                j = (i+1) % len(tour)
                delta = distance_matrix[tour[i], v] + distance_matrix[v, tour[j]] - distance_matrix[tour[i], tour[j]]
                if delta < best_delta:
                    best_delta = delta
                    best_city = v
                    best_pos = j  # insert before j
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour)

    best_tour = tour.copy()
    current_cost = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    best_cost = current_cost
    report_best_tour(best_tour)

    # Simulated annealing with 2-opt
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