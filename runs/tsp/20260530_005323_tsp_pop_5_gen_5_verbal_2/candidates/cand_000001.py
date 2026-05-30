def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    # initial tour: two farthest cities
    max_dist = -1
    best_pair = (0,1)
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i][j]
            if d > max_dist:
                max_dist = d
                best_pair = (i,j)
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
            second_best = sorted_costs[1] if len(sorted_costs)>1 else float('inf')
            regret = second_best - best
            if regret > best_regret:
                best_regret = regret
                best_city = k
                best_pos = costs.index(best)
        tour.insert(best_pos+1, best_city)
        unvisited.remove(best_city)
    tour_arr = np.array(tour)
    # helper for total distance
    def tour_dist(t):
        d = 0.0
        for idx in range(n):
            d += distance_matrix[t[idx]][t[(idx+1)%n]]
        return d
    best_dist = tour_dist(tour_arr)
    report_best_tour(tour_arr)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        L = len(tour)
        for i in range(L):
            for j in range(i+2, L):
                a = tour[i]
                b = tour[(i+1)%L]
                c = tour[j%L]
                d = tour[(j+1)%L]
                old_dist = distance_matrix[a][b] + distance_matrix[c][d]
                new_dist = distance_matrix[a][c] + distance_matrix[b][d]
                if new_dist < old_dist:
                    # reverse segment i+1..j
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    new_tour_arr = np.array(tour)
                    new_tour_dist = tour_dist(new_tour_arr)
                    if new_tour_dist < best_dist:
                        best_dist = new_tour_dist
                        report_best_tour(new_tour_arr)
                    improved = True
    return np.array(tour)