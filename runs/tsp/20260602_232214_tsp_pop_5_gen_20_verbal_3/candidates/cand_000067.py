import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    rng = np.random.default_rng()
    # nearest neighbor from random start
    start = rng.integers(n)
    tour = [start]
    unvisited = set(range(n)) - {start}
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda city: distance_matrix[cur, city])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    tour = np.array(tour)
    report_best_tour(tour)

    def tour_cost(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

    best_tour = tour.copy()
    best_cost = tour_cost(tour)
    current_tour = tour.copy()
    current_cost = best_cost

    T0 = current_cost * 0.2
    T = T0
    alpha = 0.99
    epsilon = 1e-3
    max_iter = n * 10
    while T > epsilon:
        for _ in range(max_iter):
            i = rng.integers(0, n-2)
            j = rng.integers(i+2, n)
            a, b, c, d = current_tour[i], current_tour[i+1], current_tour[j], current_tour[(j+1)%n]
            delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
            if delta < 0 or rng.random() < np.exp(-delta/T):
                current_tour = np.concatenate([current_tour[:i+1], current_tour[i+1:j+1][::-1], current_tour[j+1:]])
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = current_tour.copy()
                    report_best_tour(best_tour)
        T *= alpha
    return best_tour