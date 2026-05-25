import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)
    
    # Helper to compute tour length
    def tour_length(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))
    
    # Farthest-insertion construction
    # Find farthest pair
    max_dist = -1
    start_i = start_j = 0
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i,j]
            if d > max_dist:
                max_dist = d
                start_i, start_j = i, j
    tour = [start_i, start_j]
    remaining = set(range(n)) - {start_i, start_j}
    
    while remaining:
        # Find farthest city from current tour
        farthest_dist = -1
        farthest_city = None
        for city in remaining:
            min_dist_to_tour = min(distance_matrix[city, tcity] for tcity in tour)
            if min_dist_to_tour > farthest_dist:
                farthest_dist = min_dist_to_tour
                farthest_city = city
            elif min_dist_to_tour == farthest_dist and random.random() < 0.5:
                farthest_city = city
        # Find best insertion position
        best_increase = float('inf')
        best_pos = 0
        for pos in range(len(tour)):
            a = tour[pos]
            b = tour[(pos+1)%len(tour)]
            increase = distance_matrix[a, farthest_city] + distance_matrix[farthest_city, b] - distance_matrix[a, b]
            if increase < best_increase:
                best_increase = increase
                best_pos = pos
        tour.insert(best_pos+1, farthest_city)
        remaining.remove(farthest_city)
    
    initial_tour = np.array(tour)
    best_tour = initial_tour.copy()
    best_length = tour_length(best_tour)
    report_best_tour(best_tour)
    
    # 2-opt improvement with budget
    # We'll perform up to 'budget' 2-opt attempts (random pairs)
    attempts = 0
    improved = True
    while attempts < budget and improved:
        improved = False
        # Generate random order of (i,j) pairs to avoid bias
        indices = list(range(n))
        random.shuffle(indices)
        for i_idx in range(n-1):
            i = indices[i_idx]
            for j_idx in range(i_idx+2, n):
                j = indices[j_idx]
                if j == i+1 or (i == 0 and j == n-1):
                    continue
                # Ensure i < j for reversing segment
                if i > j:
                    i, j = j, i
                # Check if swapping edges improves
                a = best_tour[i]
                b = best_tour[(i+1)%n]
                c = best_tour[j]
                d = best_tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                if new < old:
                    # Perform swap: reverse segment from i+1 to j
                    best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                    new_len = tour_length(best_tour)
                    if new_len < best_length:
                        best_length = new_len
                        report_best_tour(best_tour.copy())
                    improved = True
                    attempts += 1
                    break
            if improved:
                break
        if not improved:
            break
    return best_tour