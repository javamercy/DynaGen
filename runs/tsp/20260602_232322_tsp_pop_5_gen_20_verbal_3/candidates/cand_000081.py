import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = float('inf')
    num_restarts = min(3, n)
    for start_city in random.sample(range(n), num_restarts):
        tour = [start_city]
        visited = {start_city}
        current = start_city
        while len(tour) < n:
            next_city = min((j for j in range(n) if j not in visited), key=lambda j: distance_matrix[current, j])
            tour.append(next_city)
            visited.add(next_city)
            current = next_city
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1) % n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)
        # Single improvement pass
        for _ in range(1):
            improved = False
            for i in range(n):
                for L in (1, 2, 3):
                    seg_indices = set((i + k) % n for k in range(L))
                    if i + L <= n:
                        segment = tour[i:i+L]
                    else:
                        segment = tour[i:] + tour[:i+L-n]
                    pred_i = (i - 1) % n
                    succ_seg = (i + L) % n
                    first_seg = segment[0]
                    last_seg = segment[-1]
                    for j in range(-1, n-1):
                        if j == -1:
                            city_j = tour[-1]
                            j_plus1 = 0
                        else:
                            if j in seg_indices or j == pred_i:
                                continue
                            city_j = tour[j]
                            j_plus1 = (j + 1) % n
                        cost_old = distance_matrix[tour[pred_i], first_seg] + distance_matrix[last_seg, tour[succ_seg]] + distance_matrix[city_j, tour[j_plus1]]
                        cost_new = distance_matrix[tour[pred_i], last_seg] + distance_matrix[city_j, first_seg] + distance_matrix[last_seg, tour[j_plus1]]
                        delta = cost_new - cost_old
                        if delta < -1e-10:
                            new_tour = [tour[k] for k in range(n) if k not in seg_indices]
                            if j == -1:
                                insertion_index = 0
                            else:
                                insertion_index = new_tour.index(city_j) + 1
                            new_tour = new_tour[:insertion_index] + segment + new_tour[insertion_index:]
                            tour = new_tour
                            dist += delta
                            if dist < best_dist - 1e-10:
                                best_dist = dist
                                best_tour = np.array(tour)
                                report_best_tour(best_tour)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    return best_tour