import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def tour_cost(tour):
        total = 0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i+1)%n]]
        return total

    # cheapest insertion heuristic
    def cheapest_insertion():
        # start with city 0
        tour = [0]
        unvisited = set(range(1, n))
        while unvisited:
            best_cost = np.inf
            best_city = -1
            best_pos = -1
            for city in unvisited:
                for pos in range(len(tour)+1):
                    # insertion cost between tour[pos-1] and tour[pos%len]
                    if pos == 0:
                        prev = tour[-1]
                        nxt = tour[0]
                    elif pos == len(tour):
                        prev = tour[-1]
                        nxt = tour[0]
                    else:
                        prev = tour[pos-1]
                        nxt = tour[pos]
                    delta = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                    if delta < best_cost:
                        best_cost = delta
                        best_city = city
                        best_pos = pos
            # insert best_city at best_pos
            tour.insert(best_pos, best_city)
            unvisited.remove(best_city)
        return np.array(tour)

    tour = cheapest_insertion()
    current_cost = tour_cost(tour)
    best_tour = tour.copy()
    best_cost = current_cost
    report_best_tour(tour)

    # simulated annealing parameters
    T0 = current_cost * 0.2
    if T0 == 0:
        T0 = 1
    T = T0
    alpha = 0.999
    epsilon = 1e-4
    max_iters_per_temp = n * 100

    rng = np.random.default_rng()

    while T > epsilon:
        for _ in range(max_iters_per_temp):
            i = rng.integers(0, n-2)
            j = rng.integers(i+2, n)
            a = tour[i]
            b = tour[i+1]
            c = tour[j]
            d = tour[(j+1)%n]
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