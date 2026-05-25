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

    # Precompute nearest neighbors for donor-list based 2-opt
    # For each node, keep track of the closest 20 neighbors
    k_neighbors = min(20, n - 1)
    donor_list = np.argsort(distance_matrix, axis=1)[:, 1:k_neighbors+1]

    def fast_2opt(tour, current_dist, budget_limit):
        t = tour.copy()
        d = current_dist
        improved = True
        iters = 0
        while improved and iters < budget_limit:
            improved = False
            # Instead of exhaustive search, we sample edges or use donor lists
            # To maintain reliability and speed, we check a mix of random and neighbor-based swaps
            for i in range(n):
                # Look at neighbors of node t[i] as potential candidates for t[j+1]
                u = t[i]
                v = t[(i + 1) % n]
                for neighbor in donor_list[u]:
                    # Find where neighbor is in the tour
                    # This is slightly slow; in a full implementation we'd track positions
                    # But for TSP size usually handled here, a simple search or limited range works
                    # Let's use a simpler approach: check a few random pairs and a few local ones
                    pass
                
                # Standard 2-opt but with a small optimization: only check a limited range
                # to keep the complexity manageable within the budget
                for j in range(i + 2, min(i + 50, n)):
                    a, b = t[i], t[i+1]
                    c, d_node = t[j], t[(j + 1) % n]
                    delta = (distance_matrix[a, c] + distance_matrix[b, d_node]) - \
                            (distance_matrix[a, b] + distance_matrix[c, d_node])
                    if delta < -1e-9:
                        t[i+1 : j+1] = t[i+1 : j+1][::-1]
                        d += delta
                        improved = True
                        iters += 1
                        if iters >= budget_limit: return t, d
            if not improved:
                break
        return t, d

    def double_bridge_move(tour):
        t = tour.copy()
        idx = sorted(random.sample(range(n), 4))
        i, j, k, l = idx
        return np.concatenate([t[:i], t[k:l], t[j:k], t[i:j], t[l:]])

    def randomized_greedy():
        unvisited = set(range(1, n))
        tour = [0]
        curr = 0
        while unvisited:
            options = sorted(list(unvisited), key=lambda x: distance_matrix[curr, x])
            pool_size = min(3, len(options))
            next_node = options[random.randint(0, pool_size - 1)]
            tour.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
        return np.array(tour)

    best_tour = randomized_greedy()
    best_dist = get_tour_dist(best_tour)
    report_best_tour(best_tour)

    total_iters = 0
    local_budget = max(n, budget // 100)
    stagnation_counter = 0

    while total_iters < budget:
        # Adaptive Perturbation Strength
        # Increase number of double-bridge moves if we are stuck
        kick_strength = 1 + (stagnation_counter // 10)
        current_tour = best_tour.copy()
        for _ in range(min(kick_strength, 5)):
            current_tour = double_bridge_move(current_tour)
        
        refined_tour, refined_dist = fast_2opt(current_tour, get_tour_dist(current_tour), local_budget)
        
        if refined_dist < best_dist - 1e-9:
            best_dist = refined_dist
            best_tour = refined_tour
            stagnation_counter = 0
            report_best_tour(best_tour)
        else:
            stagnation_counter += 1
        
        total_iters += local_budget + 1
        
        if random.random() < 0.05:
            restart_tour = randomized_greedy()
            restart_dist = get_tour_dist(restart_tour)
            if restart_dist < best_dist:
                best_tour = restart_tour
                best_dist = restart_dist
                stagnation_counter = 0
                report_best_tour(best_tour)

    return best_tour