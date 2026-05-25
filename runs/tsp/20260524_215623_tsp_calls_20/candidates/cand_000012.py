import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    # farthest pair
    max_dist = -1
    start = 0
    second = 1
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i, j]
            if d > max_dist:
                max_dist = d
                start, second = i, j
    tour = [start, second]
    visited = {start, second}
    dist = distance_matrix[start, second]
    budget_used = 0
    
    # regret insertion with tie-breaking by insertion cost
    while budget_used < budget and len(tour) < n:
        best_regret = -1.0
        best_city = -1
        best_pos = -1
        best_inc = None
        for city in range(n):
            if city in visited:
                continue
            m = len(tour)
            incs = np.zeros(m)
            for i in range(m):
                a = tour[i]
                b = tour[(i+1)%m]
                incs[i] = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
            sorted_incs = np.sort(incs)
            regret = sorted_incs[1] - sorted_incs[0]
            inc_min = incs[np.argmin(incs)]
            if regret > best_regret or (abs(regret - best_regret) < 1e-12 and (best_inc is None or inc_min < best_inc)):
                best_regret = regret
                best_city = city
                best_pos = int(np.argmin(incs))
                best_inc = inc_min
        if best_city == -1:
            break
        tour.insert(best_pos+1, best_city)
        visited.add(best_city)
        dist += best_inc
        budget_used += 1
    
    # complete tour if not full
    if len(tour) < n:
        remaining = [c for c in range(n) if c not in visited]
        np.random.shuffle(remaining)
        tour += remaining
        # recompute distance
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1)%n]]
    
    best_tour = np.array(tour)
    best_dist = dist
    report_best_tour(best_tour)
    
    # steepest descent 2-opt
    def improve_2opt(tour_arr, start_dist):
        nonlocal budget_used
        improved = True
        while budget_used < budget and improved:
            improved = False
            best_delta = 0.0
            best_i = -1
            best_j = -1
            for i in range(n-1):
                for j in range(i+2, n):
                    a = tour_arr[i]
                    b = tour_arr[(i+1)%n]
                    c = tour_arr[j]
                    d = tour_arr[(j+1)%n]
                    delta = -distance_matrix[a,b] - distance_matrix[c,d] + distance_matrix[a,c] + distance_matrix[b,d]
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_delta < -1e-12:
                i, j = best_i, best_j
                new_tour = tour_arr.copy()
                new_tour[i+1:j+1] = tour_arr[i+1:j+1][::-1]
                tour_arr = new_tour
                start_dist += best_delta
                budget_used += 1
                improved = True
                if start_dist < best_dist:
                    best_dist = start_dist
                    best_tour = tour_arr.copy()
                    report_best_tour(best_tour)
        return tour_arr, start_dist
    
    best_tour, best_dist = improve_2opt(best_tour, best_dist)
    
    # ILS with double-bridge perturb and steepest descent
    current_tour = best_tour.copy()
    current_dist = best_dist
    while budget_used < budget:
        # double-bridge perturbation
        cuts = sorted(np.random.choice(range(1, n), size=3, replace=False))
        seg1 = current_tour[:cuts[0]]
        seg2 = current_tour[cuts[0]:cuts[1]]
        seg3 = current_tour[cuts[1]:cuts[2]]
        seg4 = current_tour[cuts[2]:]
        perturbed = np.concatenate([seg1, seg3, seg2, seg4])
        budget_used += 1
        if budget_used >= budget:
            break
        # compute distance
        d_pert = 0.0
        for k in range(n):
            d_pert += distance_matrix[perturbed[k], perturbed[(k+1)%n]]
        # steepest descent on perturbed
        perturbed, d_pert = improve_2opt(perturbed, d_pert)
        if d_pert < best_dist:
            best_dist = d_pert
            best_tour = perturbed.copy()
            report_best_tour(best_tour)
        current_tour = perturbed
        current_dist = d_pert
    
    return best_tour