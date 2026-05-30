import numpy as np
import time

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0, 1])
    
    # Regret-2 construction
    best_pair = (0, 1)
    max_dist = distance_matrix[0][1]
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i][j]
            if d > max_dist:
                max_dist = d
                best_pair = (i, j)
    tour = list(best_pair)
    unvisited = set(range(n)) - set(tour)
    
    while unvisited:
        best_regret = -1.0
        best_city = None
        best_pos = None
        for k in unvisited:
            costs = []
            L = len(tour)
            for pos in range(L):
                i = tour[pos]
                j = tour[(pos+1) % L]
                cost = distance_matrix[i][k] + distance_matrix[k][j] - distance_matrix[i][j]
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
    
    tour_arr = np.array(tour)
    
    def tour_dist(t):
        d = 0.0
        for idx in range(n):
            d += distance_matrix[t[idx]][t[(idx+1)%n]]
        return d
    
    best_tour = tour_arr.copy()
    best_dist = tour_dist(best_tour)
    report_best_tour(best_tour)
    
    start_time = time.time()
    max_time = 55.0  # leave 5s margin
    
    improved = True
    while improved:
        improved = False
        # random order of i,j pairs
        indices = list(range(n))
        np.random.shuffle(indices)
        for i in indices:
            for j in indices:
                if j <= i+1 or j >= n-1:
                    continue
                # ensure i < j and not adjacent
                if i > j:
                    i, j = j, i
                if j == i+1:
                    continue
                a = tour[i]
                b = tour[(i+1) % n]
                c = tour[j]
                d = tour[(j+1) % n]
                old = distance_matrix[a][b] + distance_matrix[c][d]
                new = distance_matrix[a][c] + distance_matrix[b][d]
                if new < old - 1e-10:
                    # apply move
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    new_tour = np.array(tour)
                    new_dist = tour_dist(new_tour)
                    if new_dist < best_dist - 1e-10:
                        best_dist = new_dist
                        best_tour = new_tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    # break to restart passes
                    break
            if improved:
                break
        if improved:
            continue
        # no improvement in full pass, optionally apply double-bridge perturbation if time allows
        if time.time() - start_time > max_time:
            break
        # apply double-bridge (small perturbation) to escape local optimum
        L = n
        p1 = np.random.randint(1, L//3)
        p2 = p1 + np.random.randint(1, L//3)
        p3 = p2 + np.random.randint(1, L//3)
        segment1 = tour[:p1]
        segment2 = tour[p1:p2]
        segment3 = tour[p2:p3]
        segment4 = tour[p3:]
        tour = list(segment1) + list(segment3) + list(segment2) + list(segment4)
        new_tour = np.array(tour)
        new_dist = tour_dist(new_tour)
        if new_dist < best_dist - 1e-10:
            best_dist = new_dist
            best_tour = new_tour.copy()
            report_best_tour(best_tour)
        improved = True  # will restart 2-opt
        if time.time() - start_time > max_time:
            break
    
    return best_tour