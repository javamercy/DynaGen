import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def nn(start=0):
        tour = np.empty(n, dtype=int)
        visited = np.zeros(n, bool)
        tour[0] = start
        visited[start] = True
        cur = start
        for k in range(1, n):
            best = -1
            best_d = np.inf
            for v in range(n):
                if not visited[v]:
                    d = distance_matrix[cur, v]
                    if d < best_d:
                        best_d = d
                        best = v
            tour[k] = best
            visited[best] = True
            cur = best
        return tour

    def cost(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i+1)%n]]
        return total

    tour = nn(0)
    current_cost = cost(tour)
    best_tour = tour.copy()
    best_cost = current_cost
    report_best_tour(tour)

    rng = np.random.default_rng()
    T0 = current_cost * 0.1
    if T0 == 0:
        T0 = 1
    T = T0
    alpha = 0.99
    max_steps = n * 200
    no_improve = 0
    restart_threshold = n * 20

    for step in range(max_steps):
        i = rng.integers(0, n-2)
        j = rng.integers(i+2, n)
        a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
        delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
        if delta < 0 or rng.random() < np.exp(-delta/T):
            tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = tour.copy()
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        T *= alpha
        if no_improve >= restart_threshold:
            start_city = rng.integers(0, n)
            tour = nn(start_city)
            current_cost = cost(tour)
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = tour.copy()
                report_best_tour(best_tour)
            T = T0
            no_improve = 0

    return best_tour