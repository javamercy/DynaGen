def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    
    def tour_dist(tour):
        d = 0.0
        for idx in range(n):
            d += distance_matrix[tour[idx]][tour[(idx+1)%n]]
        return d
    
    def two_opt(tour):
        improved = True
        best_tour = tour[:]
        best_dist = tour_dist(best_tour)
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = best_tour[i]
                    b = best_tour[(i+1)%n]
                    c = best_tour[j%n]
                    d = best_tour[(j+1)%n]
                    old_dist = distance_matrix[a][b] + distance_matrix[c][d]
                    new_dist = distance_matrix[a][c] + distance_matrix[b][d]
                    if new_dist < old_dist:
                        best_tour[i+1:j+1] = reversed(best_tour[i+1:j+1])
                        new_dist_full = best_dist - old_dist + new_dist
                        best_dist = new_dist_full
                        improved = True
                        # note: we update best_dist directly
                        # but we need to recompute? simpler: recompute after all? we can keep delta
                        # for safety, recompute full distance after each improvement (small n)
                        best_dist = tour_dist(best_tour)
        return best_tour, best_dist
    
    # initial tour by regret insertion
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
    
    best_tour = tour[:]
    best_dist = tour_dist(best_tour)
    report_best_tour(np.array(best_tour))
    
    # 2-opt improvement on initial
    best_tour, best_dist = two_opt(best_tour)
    report_best_tour(np.array(best_tour))
    
    # iterated local search with double-bridge perturbation
    for _ in range(20):
        # double-bridge perturbation: cut at four random points and rejoin differently
        if n < 8:
            break  # too small for double bridge
        indices = sorted(np.random.choice(range(1, n-1), 4, replace=False))
        a, b, c, d = indices
        # new tour: [0:a] + [c:d] + [b:c] + [a:b] + [d:]
        perturbed = best_tour[:a] + best_tour[c:d] + best_tour[b:c] + best_tour[a:b] + best_tour[d:]
        # ensure it's a valid permutation (no duplicates, length n)
        if len(set(perturbed)) != n:
            continue
        # apply 2-opt
        new_tour, new_dist = two_opt(perturbed)
        if new_dist < best_dist:
            best_dist = new_dist
            best_tour = new_tour
            report_best_tour(np.array(best_tour))
        else:
            # optionally accept if not too worse (simulated annealing) but we keep simple: only better
            pass
    return np.array(best_tour)