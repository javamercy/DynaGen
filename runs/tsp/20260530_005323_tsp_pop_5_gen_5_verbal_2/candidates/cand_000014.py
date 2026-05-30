import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    
    def tour_dist(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))
    
    # Initialize with two farthest cities
    max_dist = -1
    best_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i, j]
            if d > max_dist:
                max_dist = d
                best_pair = (i, j)
    tour = list(best_pair)
    unvisited = set(range(n)) - set(tour)
    
    # Regret insertion construction
    while unvisited:
        best_city = None
        best_pos = None
        best_regret = -1
        for k in unvisited:
            costs = []
            L = len(tour)
            for pos in range(L):
                i = tour[pos]
                j = tour[(pos+1)%L]
                cost = distance_matrix[i, k] + distance_matrix[k, j] - distance_matrix[i, j]
                costs.append(cost)
            sorted_costs = sorted(costs)
            best_cost = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs) > 1 else float('inf')
            regret = second_best - best_cost
            if regret > best_regret:
                best_regret = regret
                best_city = k
                best_pos = costs.index(best_cost)
        tour.insert(best_pos+1, best_city)
        unvisited.remove(best_city)
    
    best_tour = np.array(tour)
    best_dist = tour_dist(best_tour)
    report_best_tour(best_tour)
    
    # 2-opt improvement
    def improve_2opt(tour_arr):
        nonlocal best_tour, best_dist
        improved = True
        while improved:
            improved = False
            L = len(tour_arr)
            for i in range(L):
                for j in range(i+2, L):
                    a = tour_arr[i]
                    b = tour_arr[(i+1)%L]
                    c = tour_arr[j%L]
                    d = tour_arr[(j+1)%L]
                    old_dist = distance_matrix[a, b] + distance_matrix[c, d]
                    new_dist = distance_matrix[a, c] + distance_matrix[b, d]
                    if new_dist < old_dist:
                        # Reverse segment i+1..j
                        tour_arr[i+1:j+1] = tour_arr[i+1:j+1][::-1]
                        new_dist_full = tour_dist(tour_arr)
                        if new_dist_full < best_dist:
                            best_dist = new_dist_full
                            best_tour = tour_arr.copy()
                            report_best_tour(best_tour)
                        improved = True
        return tour_arr
    
    current = best_tour.copy()
    improve_2opt(current)
    
    # Perturbation loop
    for _ in range(5):
        # Double-bridge perturbation on best_tour
        t = best_tour.tolist()
        L = len(t)
        if L < 4:
            break
        indices = sorted(random.sample(range(L), 4))
        a, b, c, d = indices
        # Build perturbed tour
        perturbed = t[:a] + t[c:d] + t[b:c] + t[a:b] + t[d:]
        perturbed_arr = np.array(perturbed)
        # Apply 2-opt
        improve_2opt(perturbed_arr)
    
    return best_tour