import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def cost(tour):
        total = 0.0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i + 1) % n]]
        return total

    def improve_sa(init_tour, global_best_cost, global_best_tour):
        tour = init_tour.copy()
        cur_cost = cost(tour)
        best_local = cur_cost
        best_local_tour = tour.copy()
        T = max(1e-8, cur_cost * 0.01)
        alpha = 0.995
        epsilon = 1e-8
        max_iters = n * 20
        while T > epsilon:
            for _ in range(max_iters):
                i = np.random.randint(0, n - 2)
                j = np.random.randint(i + 2, n)
                a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                delta = (distance_matrix[a,c] + distance_matrix[b,d] - 
                         distance_matrix[a,b] - distance_matrix[c,d])
                if delta < 0 or np.random.rand() < np.exp(-delta / T):
                    tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
                    cur_cost += delta
                    if cur_cost < best_local:
                        best_local = cur_cost
                        best_local_tour = tour.copy()
                        if cur_cost < global_best_cost[0]:
                            global_best_cost[0] = cur_cost
                            global_best_tour[0] = best_local_tour.copy()
                            report_best_tour(global_best_tour[0])
            T *= alpha
        return best_local_tour, best_local

    def nn_tour(start=0):
        tour = [start]
        visited = {start}
        current = start
        for _ in range(n-1):
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

    best_tour = nn_tour(0)
    best_cost = cost(best_tour)
    report_best_tour(best_tour)

    global_best_cost = [best_cost]
    global_best_tour = [best_tour.copy()]

    improve_sa(best_tour, global_best_cost, global_best_tour)

    num_restarts = min(5, n)
    for _ in range(num_restarts):
        start_city = np.random.randint(n)
        init_tour = nn_tour(start_city)
        improve_sa(init_tour, global_best_cost, global_best_tour)

    return global_best_tour[0]