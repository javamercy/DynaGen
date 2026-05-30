import numpy as np
import time

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0, 1])
    
    # Regret insertion construction
    max_dist = -1
    best_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i][j]
            if d > max_dist:
                max_dist = d
                best_pair = (i, j)
    tour = list(best_pair)
    unvisited = set(range(n)) - set(tour)
    while unvisited:
        best_regret = -1
        best_city = None
        best_pos = None
        for k in unvisited:
            costs = []
            L = len(tour)
            for pos in range(L):
                i = tour[pos]
                j = tour[(pos+1)%L]
                cost = distance_matrix[i][k] + distance_matrix[k][j] - distance_matrix[i][j]
                costs.append(cost)
            sorted_costs = sorted(costs)
            best = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs) > 1 else float('inf')
            regret = second_best - best
            if regret > best_regret:
                best_regret = regret
                best_city = k
                best_pos = costs.index(best)
        tour.insert(best_pos+1, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour, dtype=np.int32)
    
    # Helper total distance
    def tour_dist(t):
        d = 0.0
        for idx in range(n):
            d += distance_matrix[t[idx]][t[(idx+1)%n]]
        return d
    
    best_dist = tour_dist(tour)
    report_best_tour(tour)
    
    # 2-opt local search with timeout
    start_time = time.time()
    timeout = 55.0
    improved = True
    while improved:
        improved = False
        for i in range(n-1):
            if time.time() - start_time > timeout:
                break
            for j in range(i+1, n):
                if time.time() - start_time > timeout:
                    break
                # check if reversing segment (i, j] improves
                a = tour[i]
                b = tour[(i+1)%n]
                c = tour[j]
                d = tour[(j+1)%n]
                old_dist = distance_matrix[a][b] + distance_matrix[c][d]
                new_dist = distance_matrix[a][c] + distance_matrix[b][d]
                if new_dist + 1e-12 < old_dist:
                    # reverse segment from i+1 to j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    new_dist_full = tour_dist(tour)
                    if new_dist_full < best_dist:
                        best_dist = new_dist_full
                        report_best_tour(tour)
                    break  # restart after change
            if improved or time.time() - start_time > timeout:
                break
        if time.time() - start_time > timeout:
            break
    return tour