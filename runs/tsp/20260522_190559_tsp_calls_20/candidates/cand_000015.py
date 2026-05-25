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

    # Candidate lists for pruning the search space
    k_neighbors = 25 if n >= 80 else n - 1
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        idx = np.argsort(dists)[:k_neighbors + 1]
        candidates.append(set(idx))

    def randomized_greedy():
        unvisited = set(range(1, n))
        tour = [0]
        curr = 0
        while unvisited:
            # Pick from top 3 nearest neighbors for variety
            options = sorted(list(unvisited), key=lambda x: distance_matrix[curr, x])
            pool_size = min(3, len(options))
            next_node = options[random.randint(0, pool_size - 1)]
            tour.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
        return np.array(tour)

    def fast_2opt(tour, current_dist, local_budget):
        t = tour.copy()
        d = current_dist
        pos_map = np.zeros(n, dtype=int)
        for i in range(n): pos_map[t[i]] = i
        
        iters = 0
        improved = True
        while improved and iters < local_budget:
            improved = False
            for i in range(n):
                if iters >= local_budget: break
                u = t[i]
                v = t[(i + 1) % n]
                
                # Use candidate lists to find a better edge
                for x in candidates[u]:
                    if x == u or x == v: continue
                    px = pos_map[x]
                    y = t[(px + 1) % n]
                    if y == u or y == v: continue
                    
                    delta = (distance_matrix[u, x] + distance_matrix[v, y]) - \
                            (distance_matrix[u, v] + distance_matrix[x, y])
                    
                    if delta < -1e-9:
                        # Reverse segment from i+1 to px
                        idx1, idx2 = (i + 1) % n, px
                        t_list = t.tolist()
                        if idx1 <= idx2:
                            t_list[idx1 : idx2 + 1] = t_list[idx2 : idx1 - 1 if idx1 > 0 else None : -1]
                        else:
                            segment = t_list[idx1:] + t_list[:idx2 + 1]
                            rev = segment[::-1]
                            for j in range(len(rev)):
                                t_list[(idx1 + j) % n] = rev[j]
                        
                        t = np.array(t_list)
                        for k in range(n): pos_map[t[k]] = k
                        d += delta
                        improved = True
                        iters += 1
                        break
                if improved: break
        return t, d

    # Initial Solution
    best_tour = randomized_greedy()
    best_dist = get_tour_dist(best_tour)
    report_best_tour(best_tour)

    total_iters = 0
    while total_iters < budget:
        # Perturbation (Kick)
        current_tour = best_tour.copy()
        if total_iters > 0:
            # Simple segment reversal (part of a 4-opt double-bridge)
            i, j = sorted(random.sample(range(n), 2))
            current_tour[i:j] = current_tour[i:j][::-1]
        
        # Local Search
        local_budget = min(n * 2, (budget - total_iters) // 5 + 1)
        refined_tour, refined_dist = fast_2opt(current_tour, get_tour_dist(current_tour), local_budget)
        
        if refined_dist < best_dist - 1e-9:
            best_dist = refined_dist
            best_tour = refined_tour
            report_best_tour(best_tour)
        
        total_iters += local_budget + 1
        
        # Periodic full restart
        if random.random() < 0.1:
            restart_tour = randomized_greedy()
            restart_dist = get_tour_dist(restart_tour)
            if restart_dist < best_dist:
                best_tour = restart_tour
                best_dist = restart_dist
                report_best_tour(best_tour)

    return best_tour