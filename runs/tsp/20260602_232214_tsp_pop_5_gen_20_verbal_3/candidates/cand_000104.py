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

    def two_opt_delta(tour, i, j):
        a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
        return distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]

    def run_sa(init_tour, budget):
        tour = init_tour.copy()
        cur_cost = cost(tour)
        best_local = cur_cost
        best_local_tour = tour.copy()
        T = np.max(distance_matrix) * 0.1
        T_end = 1e-8
        alpha = np.exp(np.log(T_end / T) / budget) if budget > 0 else 1.0
        for _ in range(budget):
            i = np.random.randint(0, n-2)
            j = np.random.randint(i+2, n)
            delta = two_opt_delta(tour, i, j)
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-12)):
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

    total_budget = max(1000, n * 200)
    num_restarts = min(5, n)
    budget_per_restart = total_budget // num_restarts

    best_tour = nn_tour(0)
    best_cost = cost(best_tour)
    report_best_tour(best_tour)
    global_best_cost = [best_cost]
    global_best_tour = [best_tour.copy()]

    run_sa(best_tour, budget_per_restart)

    for _ in range(num_restarts):
        start_city = np.random.randint(n)
        init_tour = nn_tour(start_city)
        run_sa(init_tour, budget_per_restart)

    return global_best_tour[0]