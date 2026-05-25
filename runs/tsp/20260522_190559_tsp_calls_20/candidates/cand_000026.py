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
                if improved: break
        return t, d

    def double_bridge_move(tour):
        # 4-opt move that breaks 4 edges and reconnects them differently
        t = tour.copy()
        # Pick 4 random indices
        idx = sorted(random.sample(range(n), 4))
        i, j, k, l = idx
        # The double bridge move: reverse and swap segments
        # Original: [0...i][i+1...j][j+1...k][k+1...l][l+1...n-1]
        # New: [0...i][k+1...l][j+1...k][i+1...j][l+1...n-1]
        segment1 = t[i+1 : j+1]
        segment2 = t[j+1 : k+1]
        segment3 = t[k+1 : l+1]
        t[i+1 : l+1] = np.concatenate([segment3, segment2, segment1])
        return t

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

    # Initial Solution
    best_tour = randomized_greedy()
    best_dist = get_tour_dist(best_tour)
    report_best_tour(best_tour)

    total_iters = 0
    # Adaptive local budget scaled by n
    local_budget = min(n * 5, budget // 5 + 1)

    while total_iters < budget:
        # 1. Perturbation (Kick): Double Bridge Move
        current_tour = double_bridge_move(best_tour)
        
        # 2. Local Search
        current_dist = get_tour_dist(current_tour)
        refined_tour, refined_dist = fast_2opt(current_tour, current_dist, local_budget)
        
        if refined_dist < best_dist - 1e-9:
            best_dist = refined_dist
            best_tour = refined_tour
            report_best_tour(best_tour)
        
        total_iters += local_budget + 1
        
        # Periodic restart
        if random.random() < 0.05:
            restart_tour = randomized_greedy()
            restart_dist = get_tour_dist(restart_tour)
            if restart_dist < best_dist:
                best_dist = restart_dist
                best_tour = restart_tour
                report_best_tour(best_tour)

    return best_tour