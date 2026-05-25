import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)

    def tour_length(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

    def farthest_insertion(rng):
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
            farthest_dist = -1
            farthest_city = None
            for city in remaining:
                min_dist = min(distance_matrix[city, tcity] for tcity in tour)
                if min_dist > farthest_dist:
                    farthest_dist = min_dist
                    farthest_city = city
                elif min_dist == farthest_dist and rng.random() < 0.5:
                    farthest_city = city
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
        return np.array(tour)

    # Initial tour with global seed
    global_rng = random.Random(seed)
    best_tour = farthest_insertion(global_rng)
    best_length = tour_length(best_tour)
    report_best_tour(best_tour.copy())

    if budget <= 0:
        return best_tour

    global_attempts = 0
    restart_num = 0
    while global_attempts < budget:
        restart_rng = random.Random(seed + restart_num + 1)
        tour = farthest_insertion(restart_rng)
        improved = True
        while improved and global_attempts < budget:
            improved = False
            indices = list(range(n))
            restart_rng.shuffle(indices)
            for i_idx in range(n-1):
                i = indices[i_idx]
                for j_idx in range(i_idx+2, n):
                    j = indices[j_idx]
                    if j == i+1 or (i == 0 and j == n-1):
                        continue
                    if i > j:
                        i, j = j, i
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    old = distance_matrix[a,b] + distance_matrix[c,d]
                    new = distance_matrix[a,c] + distance_matrix[b,d]
                    if new < old:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        new_len = tour_length(tour)
                        global_attempts += 1
                        if new_len < best_length:
                            best_length = new_len
                            best_tour = tour.copy()
                            report_best_tour(best_tour.copy())
                        improved = True
                        break
                if improved:
                    break
        restart_num += 1
        if restart_num > 1000:  # safety break
            break
    return best_tour