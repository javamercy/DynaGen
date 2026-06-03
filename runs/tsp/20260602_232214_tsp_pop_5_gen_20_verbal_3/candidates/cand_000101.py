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

    def two_opt_local(tour, best_cost, best_tour):
        improved = True
        current_tour = tour.copy()
        current_cost = cost(current_tour)
        while improved:
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n):
                    a = current_tour[i]
                    b = current_tour[i + 1]
                    c = current_tour[j]
                    d = current_tour[(j + 1) % n]
                    delta = (distance_matrix[a, c] + distance_matrix[b, d] -
                             distance_matrix[a, b] - distance_matrix[c, d])
                    if delta < -1e-8:
                        current_tour = np.concatenate([current_tour[:i+1],
                                                        current_tour[i+1:j+1][::-1],
                                                        current_tour[j+1:]])
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost[0]:
                            best_cost[0] = current_cost
                            best_tour[0] = current_tour.copy()
                            report_best_tour(best_tour[0])
                        break
                if improved:
                    break
        return current_tour, current_cost

    def perturbation(tour):
        # Random 2-opt move (not necessarily improving)
        i = np.random.randint(0, n - 2)
        j = np.random.randint(i + 2, n)
        return np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])

    def nn_tour(start=0):
        tour = [start]
        visited = {start}
        current = start
        for _ in range(n - 1):
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

    # Initial best from nearest neighbor with start 0
    best_tour = nn_tour(0)
    best_cost = cost(best_tour)
    report_best_tour(best_tour)

    global_best_cost = [best_cost]
    global_best_tour = [best_tour.copy()]

    # Multiple restarts
    num_restarts = min(5, max(1, n // 20))
    for restart in range(num_restarts):
        if restart == 0:
            init_tour = best_tour
        else:
            start_city = np.random.randint(n)
            init_tour = nn_tour(start_city)
        
        curr_tour, curr_cost = two_opt_local(init_tour, global_best_cost, global_best_tour)
        
        # Iterated local search: perturbation + local search
        for _ in range(10):  # perturbation iterations
            perturbed_tour = perturbation(curr_tour)
            curr_tour, curr_cost = two_opt_local(perturbed_tour, global_best_cost, global_best_tour)
            # Early stop if improvement found and we are close to best
            # optional: break if no improvement after some iterations

    return global_best_tour[0]