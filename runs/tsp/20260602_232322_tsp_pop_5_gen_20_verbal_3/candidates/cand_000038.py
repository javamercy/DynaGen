import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    # farthest insertion construction
    start = 0
    tour = [start]
    remaining = set(range(1, n))
    while remaining:
        best_city = -1
        best_min_dist = -1
        for city in remaining:
            min_dist = min(distance_matrix[city, t] for t in tour)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_city = city
        # insert best_city at best position (min cost increase)
        best_pos = 0
        best_cost_increase = float('inf')
        for pos in range(len(tour)+1):
            before = tour[pos-1] if pos > 0 else tour[-1]
            after = tour[pos] if pos < len(tour) else tour[0]
            inc = distance_matrix[before, best_city] + distance_matrix[best_city, after] - distance_matrix[before, after]
            if inc < best_cost_increase:
                best_cost_increase = inc
                best_pos = pos
        tour.insert(best_pos, best_city)
        remaining.remove(best_city)
    
    def compute_dist(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))
    
    best_tour = np.array(tour)
    best_dist = compute_dist(best_tour)
    report_best_tour(best_tour)
    
    def two_opt_best(t):
        tour_list = t.tolist() if isinstance(t, np.ndarray) else t
        improved = True
        while improved:
            best_gain = 0
            best_i = best_j = -1
            for i in range(n-1):
                for j in range(i+2, n):
                    a = tour_list[i]
                    b = tour_list[(i+1)%n]
                    c = tour_list[j]
                    d = tour_list[(j+1)%n]
                    old = distance_matrix[a,b] + distance_matrix[c,d]
                    new = distance_matrix[a,c] + distance_matrix[b,d]
                    gain = old - new
                    if gain > best_gain:
                        best_gain = gain
                        best_i, best_j = i, j
            if best_gain > 1e-10:
                new_tour = tour_list[:best_i+1] + tour_list[best_i+1:best_j+1][::-1] + tour_list[best_j+1:]
                tour_list = new_tour
            else:
                improved = False
        return np.array(tour_list)
    
    current_tour = best_tour
    max_ils = max(10, n//5)
    for _ in range(max_ils):
        # double-bridge perturbation
        points = sorted(random.sample(range(n), 4))
        a,b,c,d = points
        new_tour = np.concatenate([
            current_tour[:a+1],
            current_tour[c+1:d+1],
            current_tour[b+1:c+1],
            current_tour[a+1:b+1],
            current_tour[d+1:]
        ])
        # local search
        improved_tour = two_opt_best(new_tour)
        dist_improved = compute_dist(improved_tour)
        if dist_improved < best_dist:
            best_dist = dist_improved
            best_tour = improved_tour.copy()
            report_best_tour(best_tour)
        current_tour = improved_tour
    return best_tour