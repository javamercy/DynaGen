import numpy as np
import random

def report_best_tour(tour):
    # This is a placeholder as per the prompt's implication of tracking the best
    pass

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n)

    random.seed(seed)
    np.random.seed(seed)

    def get_dist(tour):
        d = 0
        for i in range(n):
            d += distance_matrix[tour[i], tour[(i + 1) % n]]
        return d

    # Initial Incumbent: Greedy
    curr_tour = np.zeros(n, dtype=int)
    visited = [False] * n
    curr_tour[0] = 0
    visited[0] = True
    for i in range(1, n):
        prev = curr_tour[i-1]
        best_next = -1
        min_d = float('inf')
        for j in range(n):
            if not visited[j] and distance_matrix[prev, j] < min_d:
                min_d = distance_matrix[prev, j]
                best_next = j
        curr_tour[i] = best_next
        visited[best_next] = True

    best_tour = np.copy(curr_tour)
    best_dist = get_dist(best_tour)
    report_best_tour(best_tour)

    def local_search(tour, current_dist, budget_limit):
        nonlocal best_dist, best_tour
        improved = True
        iters = 0
        while improved and iters < budget_limit:
            improved = False
            iters += 1
            for i in range(n):
                for j in range(i + 2, n):
                    # 2-opt move
                    # tour[i+1...j] reversed
                    # cost change: (i, i+1) + (j, j+1) -> (i, j) + (i+1, j+1)
                    a, b = tour[i], tour[(i + 1) % n]
                    c, d_node = tour[j], tour[(j + 1) % n]
                    diff = (distance_matrix[a, c] + distance_matrix[b, d_node]) - \n                           (distance_matrix[a, b] + distance_matrix[c, d_node])
                    if diff < -1e-9:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        current_dist += diff
                        improved = True
                        if current_dist < best_dist:
                            best_dist = current_dist
                            best_tour = np.copy(tour)
                            report_best_tour(best_tour)
                if improved: break
        return tour, current_dist

    # ILS Main Loop
    curr_dist = best_dist
    budget_per_ls = max(1, budget // 10)
    iterations = 0
    
    while iterations < budget:
        iterations += 1
        # Local Search
        curr_tour, curr_dist = local_search(curr_tour, curr_dist, budget_per_ls)
        
        # Perturbation: Double-Bridge
        # Pick 4 indices and swap 4 segments
        idx = sorted(random.sample(range(n), 4))
        i, j, k, l = idx
        # Segments: [0,i], (i,j], (j,k], (k,l], [l,n-1]
        # New: [0,i], (k,l], (j,k], (i,j], [l,n-1]
        new_tour = np.concatenate([
            curr_tour[:i+1],
            curr_tour[k+1:l+1],
            curr_tour[j+1:k+1],
            curr_tour[i+1:j+1],
            curr_tour[l+1:]
        ])
        curr_tour = new_tour
        curr_dist = get_dist(curr_tour)

    return best_tour