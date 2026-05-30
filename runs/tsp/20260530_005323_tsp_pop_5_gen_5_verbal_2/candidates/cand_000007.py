import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0, 1])
    
    #------- Construction: regret insertion -------
    # start with two farthest cities
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
            best_cost = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs)>1 else float('inf')
            regret = second_best - best_cost
            if regret > best_regret:
                best_regret = regret
                best_city = k
                best_pos = costs.index(best_cost)  # position of best insertion
        tour.insert(best_pos+1, best_city)
        unvisited.remove(best_city)
    
    tour_arr = np.array(tour)
    
    def tour_dist(t):
        d = 0.0
        for idx in range(n):
            d += distance_matrix[t[idx]][t[(idx+1)%n]]
        return d
    
    best_dist = tour_dist(tour_arr)
    report_best_tour(tour_arr)
    
    #------- Local search: 2-opt -------
    improved = True
    while improved:
        improved = False
        L = n
        for i in range(L):
            for j in range(i+2, L):
                a = tour[i]
                b = tour[(i+1)%L]
                c = tour[j%L]
                d = tour[(j+1)%L]
                old = distance_matrix[a][b] + distance_matrix[c][d]
                new = distance_matrix[a][c] + distance_matrix[b][d]
                if new < old - 1e-10:
                    # reverse segment i+1..j
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    new_tour = np.array(tour)
                    new_dist = tour_dist(new_tour)
                    if new_dist < best_dist - 1e-10:
                        best_dist = new_dist
                        report_best_tour(new_tour)
                    improved = True
    
    #------- Local search: Swap (relocate) -------
    improved_swap = True
    while improved_swap:
        improved_swap = False
        L = n
        for i in range(L):
            for j in range(i+1, L):
                if j == i: continue
                # swap city at i and j
                tour[i], tour[j] = tour[j], tour[i]
                new_tour = np.array(tour)
                new_dist = tour_dist(new_tour)
                if new_dist < best_dist - 1e-10:
                    best_dist = new_dist
                    report_best_tour(new_tour)
                    improved_swap = True
                else:
                    # revert swap
                    tour[i], tour[j] = tour[j], tour[i]
    
    #------- Additional 2-opt after swap -------
    if improved_swap:
        improved = True
        while improved:
            improved = False
            L = n
            for i in range(L):
                for j in range(i+2, L):
                    a = tour[i]
                    b = tour[(i+1)%L]
                    c = tour[j%L]
                    d = tour[(j+1)%L]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new = distance_matrix[a][c] + distance_matrix[b][d]
                    if new < old - 1e-10:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        new_tour = np.array(tour)
                        new_dist = tour_dist(new_tour)
                        if new_dist < best_dist - 1e-10:
                            best_dist = new_dist
                            report_best_tour(new_tour)
                        improved = True
    
    return np.array(tour)