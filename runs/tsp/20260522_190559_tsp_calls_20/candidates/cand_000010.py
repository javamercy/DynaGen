import numpy as np
import random

def report_best_tour(tour):
    pass

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 1:
        return np.arange(n)
    if n == 2:
        return np.array([0, 1])

    random.seed(seed)
    np.random.seed(seed)

    def get_tour_dist(t):
        d = 0
        for i in range(n):
            d += distance_matrix[t[i], t[(i + 1) % n]]
        return d

    def fast_2opt(tour, current_dist, budget_limit):
        t = tour.copy()
        d = current_dist
        improved = True
        iters = 0
        while improved and iters < budget_limit:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    # Edge (i, i+1) and (j, j+1)
                    a, b = t[i], t[i+1]
                    c, d_node = t[j], t[(j + 1) % n]
                    
                    # Change in distance: remove (a,b) and (c,d_node), add (a,c) and (b,d_node)
                    delta = (distance_matrix[a, c] + distance_matrix[b, d_node]) - \
                            (distance_matrix[a, b] + distance_matrix[c, d_node])
                    
                    if delta < -1e-9:
                        t[i+1 : j+1] = t[i+1 : j+1][::-1]
                        d += delta
                        improved = True
                        iters += 1
                        if iters >= budget_limit: return t, d
                if improved: break
        return t, d

    def randomized_greedy():
        unvisited = set(range(1, n))
        tour = [0]
        curr = 0
        while unvisited:
            # Pick from top 3 nearest neighbors to introduce variety
            options = sorted(list(unvisited), key=lambda x: distance_matrix[curr, x])
            pool_size = min(3, len(options))
            next_node = options[random.randint(0, pool_size - 1)]
            tour.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
        return np.array(tour)

    # Initial Solution
    best_tour = randomized_greedy()
    best_dist = get_tour_dist(best_tour)
    report_best_tour(best_tour)

    # Budget allocation: split between restarts and local search
    # We use a simple ILS loop
    total_iters = 0
    while total_iters < budget:
        # 1. Perturbation (Kick): Randomly reverse a segment
        current_tour = best_tour.copy()
        if total_iters > 0:
            i, j = sorted(random.sample(range(n), 2))
            current_tour[i:j] = current_tour[i:j][::-1]
        
        # 2. Local Search
        # Allocate a portion of budget to 2-opt (e.g., n*2 iterations)
        local_budget = min(n * 2, budget // 10 + 1)
        refined_tour, refined_dist = fast_2opt(current_tour, get_tour_dist(current_tour), local_budget)
        
        if refined_dist < best_dist - 1e-9:
            best_dist = refined_dist
            best_tour = refined_tour
            report_best_tour(best_tour)
        
        total_iters += local_budget + 1
        
        # Periodic full restart to avoid stagnation
        if random.random() < 0.1:
            restart_tour = randomized_greedy()
            restart_dist = get_tour_dist(restart_tour)
            if restart_dist < best_dist:
                best_tour = restart_tour
                best_dist = restart_dist
                report_best_tour(best_tour)

    return best_tour