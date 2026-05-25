import numpy as np
import random

def report_best_tour(tour):
    # This is a placeholder for the external reporting system
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

    # Initial incumbent
    current_tour = np.arange(n)
    np.random.shuffle(current_tour)
    best_tour = current_tour.copy()
    best_dist = get_dist(best_tour)
    report_best_tour(best_tour)

    def local_search(tour):
        nonlocal best_dist
        improved = True
        curr_tour = tour.copy()
        curr_dist = get_dist(curr_tour)
        
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    # 2-opt swap: reverse segment [i+1, j]
                    # delta = dist(i, i+1) + dist(j, j+1) - dist(i, j) - dist(i+1, j+1)
                    a, b = curr_tour[i], curr_tour[(i + 1) % n]
                    c, d_node = curr_tour[j], curr_tour[(j + 1) % n]
                    
                    diff = (distance_matrix[a, c] + distance_matrix[b, d_node]) - \
                           (distance_matrix[a, b] + distance_matrix[c, d_node])
                    
                    if diff < -1e-9:
                        curr_tour[i+1:j+1] = curr_tour[i+1:j+1][::-1]
                        curr_dist += diff
                        improved = True
            if curr_dist < best_dist:
                best_dist = curr_dist
                return curr_tour, curr_dist
        return curr_tour, curr_dist

    def perturb(tour):
        # Double-bridge move: 4-opt move that doesn't break permutation
        new_tour = tour.copy()
        indices = sorted(random.sample(range(n), 4))
        i, j, k, l = indices
        # Split into 4 segments: [0,i], [i+1,j], [j+1,k], [k+1,l...n-1]
        # Rearrange to break 2 local minima
        seg1 = new_tour[:i+1]
        seg2 = new_tour[i+1:j+1]
        seg3 = new_tour[j+1:k+1]
        seg4 = new_tour[k+1:]
        return np.concatenate([seg1, seg3, seg2, seg4])

    # Budget-bounded ILS loop
    iterations = 0
    while iterations < budget:
        # Local Improvement
        current_tour, current_dist = local_search(current_tour)
        
        if current_dist < best_dist:
            best_tour = current_tour.copy()
            best_dist = current_dist
            report_best_tour(best_tour)
        
        # Diversification
        current_tour = perturb(current_tour)
        iterations += 1
        
    return best_tour