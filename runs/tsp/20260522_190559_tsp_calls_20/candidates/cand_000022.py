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

    def randomized_greedy():
        unvisited = set(range(1, n))
        tour = [0]
        curr = 0
        while unvisited:
            options = sorted(list(unvisited), key=lambda x: distance_matrix[curr, x])
            pool_size = min(5, len(options))
            next_node = options[random.randint(0, pool_size - 1)]
            tour.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
        return np.array(tour)

    def two_opt_move(tour, i, j):
        t = tour.copy()
        t[i+1 : j+1] = t[i+1 : j+1][::-1]
        return t

    def three_opt_move(tour):
        t = tour.copy()
        idx = sorted(random.sample(range(n), 3))
        i, j, k = idx
        # Simple 3-opt variant: reverse one segment and swap another
        # This breaks 3 edges and creates a new configuration
        seg1 = t[:i+1]
        seg2 = t[i+1:j+1][::-1]
        seg3 = t[j+1:k+1]
        seg4 = t[k+1:]
        return np.concatenate([seg1, seg3, seg2, seg4])

    def random_swap(tour):
        t = tour.copy()
        i, j = random.sample(range(n), 2)
        t[i], t[j] = t[j], t[i]
        return t

    # Initial Solution
    best_tour = randomized_greedy()
    best_dist = get_tour_dist(best_tour)
    report_best_tour(best_tour)

    iters = 0
    # VNS neighborhood index
    k_neighborhood = 1
    max_k = 3

    while iters < budget:
        improved_in_this_cycle = False
        
        # Variable Neighborhood Search Logic
        for k in range(1, max_k + 1):
            # Generate candidate based on neighborhood k
            if k == 1:
                # 2-opt local search attempt
                i, j = sorted(random.sample(range(n - 1), 2))
                candidate = two_opt_move(best_tour, i, j)
            elif k == 2:
                # 3-opt perturbation
                candidate = three_opt_move(best_tour)
            else:
                # Random swap / Shake
                candidate = random_swap(best_tour)
            
            cand_dist = get_tour_dist(candidate)
            
            # Local improvement (mini 2-opt)
            if cand_dist < best_dist * 1.1: # Only refine if promising
                # Quick 2-opt polish
                for _ in range(min(n, 10)):
                    ii, jj = sorted(random.sample(range(n - 1), 2))
                    a, b = candidate[ii], candidate[ii+1]
                    c, d_node = candidate[jj], candidate[(jj + 1) % n]
                    delta = (distance_matrix[a, c] + distance_matrix[b, d_node]) - \
                            (distance_matrix[a, b] + distance_matrix[c, d_node])
                    if delta < -1e-9:
                        candidate = two_opt_move(candidate, ii, jj)
                        cand_dist += delta

            if cand_dist < best_dist - 1e-9:
                best_dist = cand_dist
                best_tour = candidate
                report_best_tour(best_tour)
                improved_in_this_cycle = True
                break # Return to smallest neighborhood
            
            iters += 1
            if iters >= budget: break
        
        if not improved_in_this_cycle:
            # If we couldn't improve, occasionally restart with a new greedy tour
            if random.random() < 0.1:
                restart_tour = randomized_greedy()
                restart_dist = get_tour_dist(restart_tour)
                if restart_dist < best_dist:
                    best_dist = restart_dist
                    best_tour = restart_tour
                    report_best_tour(best_tour)
        
        iters += 1

    return best_tour