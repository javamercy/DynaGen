import numpy as np
import time

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0, 1])
    
    # ---------- Regret-2 construction ----------
    # start with farthest pair
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
    
    tour = np.array(tour, dtype=int)
    
    def tour_dist(t):
        d = 0.0
        for idx in range(n):
            d += distance_matrix[t[idx]][t[(idx+1)%n]]
        return d
    
    best_dist = tour_dist(tour)
    report_best_tour(tour.copy())
    
    # ---------- 2-opt first improvement with time limit ----------
    start_time = time.time()
    timeout = 55.0  # seconds
    rng = np.random.default_rng()
    improved = True
    while improved:
        improved = False
        # random order of i and j edges
        indices = list(range(n))
        rng.shuffle(indices)
        for a_idx in range(n):
            i = indices[a_idx]
            for b_idx in range(a_idx+1, n):
                jj = indices[b_idx]
                # ensure jj > i+1 (wrapping considered)
                if jj <= i+1:
                    continue
                j = jj
                if j == i+1:
                    continue
                # edge between i and i+1, and j and j+1
                a = tour[i]
                b = tour[(i+1) % n]
                c = tour[j]
                d = tour[(j+1) % n]
                old = distance_matrix[a][b] + distance_matrix[c][d]
                new = distance_matrix[a][c] + distance_matrix[b][d]
                if new < old - 1e-10:
                    # reverse segment i+1 .. j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    new_dist = tour_dist(tour)
                    if new_dist < best_dist - 1e-10:
                        best_dist = new_dist
                        report_best_tour(tour.copy())
                    improved = True
                    # check time
                    if time.time() - start_time > timeout:
                        return tour
                    break  # first improvement: restart loop
            if improved:
                break
        # if not improved after full pass, exit
    return tour