def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        return np.array(range(n))
    # farthest insertion construction
    # find the farthest pair
    i, j = np.unravel_index(np.argmax(distance_matrix), (n, n))
    tour = [i, j]
    remaining = set(range(n)) - {i, j}
    while remaining:
        # find farthest point from current tour
        dists = np.array([min(distance_matrix[c, t] for t in tour) for c in remaining])
        farthest_idx = np.argmax(dists)
        farthest = list(remaining)[farthest_idx]
        # find best insertion position
        best_inc = np.inf
        best_pos = 0
        for k in range(len(tour)):
            a = tour[k]
            b = tour[(k+1) % len(tour)]
            inc = distance_matrix[a, farthest] + distance_matrix[farthest, b] - distance_matrix[a, b]
            if inc < best_inc:
                best_inc = inc
                best_pos = k+1
        tour.insert(best_pos, farthest)
        remaining.remove(farthest)
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    # 2-opt improvement
    improved = True
    best_tour = tour_arr.copy()
    best_cost = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    while improved:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                if j == i+1:
                    continue
                new_tour = best_tour.copy()
                new_tour[i+1:j+1] = best_tour[j:i:-1]  # careful: slice reversal
                # compute new tour length difference
                # affected edges: (i, i+1) and (j, (j+1)%n) replaced by (i, j) and (i+1, j+1)
                a = best_tour[i]
                b = best_tour[i+1]
                c = best_tour[j]
                d = best_tour[(j+1)%n]
                delta = (distance_matrix[a,c] + distance_matrix[b,d]) - (distance_matrix[a,b] + distance_matrix[c,d])
                if delta < -1e-10:
                    best_tour = new_tour
                    best_cost += delta
                    improved = True
                    report_best_tour(best_tour)
        # Also consider wrapping around? 2-opt typically handles wrap by modulo indexing, already done.
    return best_tour