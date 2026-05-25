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
        """Performs a 4-opt double-bridge move to perturb the tour."""
        t = tour.copy()
        # Pick 4 random indices
        indices = sorted(random.sample(range(n), 4))
        i, j, k, l = indices
        # Split tour into 4 segments: [0..i], [i+1..j], [j+1..k], [k+1..l], [l+1..n-1]
        # Reconnect segments to create a new valid tour
        # Original: (i, i+1), (j, j+1), (k, k+1), (l, l+1)
        # New: (i, j+1), (k, i+1), (j, k+1), (l, l+1) - simplified logic below
        part1 = t[:i+1]
        part2 = t[i+1:j+1]
        part3 = t[j+1:k+1]
        part4 = t[k+1:l+1]
        part5 = t[l+1:]
        # New sequence: part1 -> part3 -> part2 -> part4 -> part5
        # This changes the topology significantly
        return np.concatenate([part1, part3, part2, part4, part5])

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
    # Adaptive local search budget based on problem size
    local_budget = max(n, min(n * 2, budget // 20))
    
    while total_iters < budget:
        # 1. Perturbation: Double-Bridge Move (4-opt)
        # This is materially different from the parent's simple segment reversal
        current_tour = double_bridge_move(best_tour)
        
        # 2. Local Search (2-opt)
        refined_tour, refined_dist = fast_2opt(current_tour, get_tour_dist(current_tour), local_budget)
        
        if refined_dist < best_dist - 1e-9:
            best_dist = refined_dist
            best_tour = refined_tour
            report_best_tour(best_tour)
        
        total_iters += local_budget + 1
        
        # Occasional full restart to explore new regions
        if random.random() < 0.05:
            restart_tour = randomized_greedy()
            restart_dist = get_tour_dist(restart_tour)
            if restart_dist < best_dist:
                best_dist = restart_dist
                best_tour = restart_tour
                report_best_tour(best_tour)

    return best_tour