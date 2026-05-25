import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = len(distance_matrix)
    
    # initial random tour
    tour = np.random.permutation(n)
    best_tour = tour.copy()
    best_dist = 0.0
    for i in range(n):
        best_dist += distance_matrix[tour[i], tour[(i+1)%n]]
    report_best_tour(best_tour)
    
    if budget <= 0 or n <= 2:
        return best_tour
    
    # farthest pair seed
    max_dist = -1
    start = 0
    second = 1
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i, j]
            if d > max_dist:
                max_dist = d
                start, second = i, j
    
    current_tour = [start, second]
    visited = {start, second}
    dist = distance_matrix[start, second]
    budget_used = 0
    
    # regret insertion with tie-breaking by insertion cost
    while budget_used < budget and len(current_tour) < n:
        best_regret = -1
        best_city = None
        best_pos = None
        best_inc = None
        for city in range(n):
            if city in visited:
                continue
            m = len(current_tour)
            incs = np.zeros(m)
            for i in range(m):
                prev = current_tour[i]
                nxt = current_tour[(i+1)%m]
                incs[i] = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
            sorted_incs = np.sort(incs)
            regret = sorted_incs[1] - sorted_incs[0]
            inc_min = incs[np.argmin(incs)]
            if regret > best_regret or (regret == best_regret and (best_inc is None or inc_min < best_inc)):
                best_regret = regret
                best_city = city
                best_pos = int(np.argmin(incs))
                best_inc = inc_min
        if best_city is None:
            break
        current_tour.insert(best_pos+1, best_city)
        visited.add(best_city)
        dist += best_inc
        budget_used += 1
    
    if len(current_tour) == n:
        candidate = np.array(current_tour)
        if dist < best_dist:
            best_dist = dist
            best_tour = candidate.copy()
            report_best_tour(best_tour)
    
    # efficient 2-opt on best_tour
    remaining = budget - budget_used
    i = 0
    while remaining > 0 and i < n - 1:
        improved = False
        j = i + 2
        while j < n and remaining > 0:
            a, b = best_tour[i], best_tour[(i+1)%n]
            c, d = best_tour[j], best_tour[(j+1)%n]
            delta = -distance_matrix[a,b] - distance_matrix[c,d] + distance_matrix[a,c] + distance_matrix[b,d]
            if delta < 0:
                new_tour = best_tour.copy()
                new_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                best_tour = new_tour
                best_dist += delta
                report_best_tour(best_tour)
                remaining -= 1
                improved = True
                break
            j += 1
        if improved:
            continue
        else:
            i += 1
    
    # ILS: perturb and re-optimize
    while remaining > 0:
        # double-bridge perturbation
        cuts = sorted(np.random.choice(range(1, n), size=3, replace=False))
        tour = best_tour.copy()
        seg1 = tour[:cuts[0]]
        seg2 = tour[cuts[0]:cuts[1]]
        seg3 = tour[cuts[1]:cuts[2]]
        seg4 = tour[cuts[2]:]
        perturbed = np.concatenate([seg1, seg3, seg2, seg4])
        # compute distance
        d_pert = 0.0
        for k in range(n):
            d_pert += distance_matrix[perturbed[k], perturbed[(k+1)%n]]
        remaining -= 1
        if remaining <= 0:
            break
        # local search on perturbed
        i = 0
        while remaining > 0 and i < n - 1:
            improved = False
            j = i + 2
            while j < n and remaining > 0:
                a, b = perturbed[i], perturbed[(i+1)%n]
                c, d = perturbed[j], perturbed[(j+1)%n]
                delta = -distance_matrix[a,b] - distance_matrix[c,d] + distance_matrix[a,c] + distance_matrix[b,d]
                if delta < 0:
                    new_pert = perturbed.copy()
                    new_pert[i+1:j+1] = perturbed[i+1:j+1][::-1]
                    perturbed = new_pert
                    d_pert += delta
                    remaining -= 1
                    improved = True
                    if d_pert < best_dist:
                        best_dist = d_pert
                        best_tour = perturbed.copy()
                        report_best_tour(best_tour)
                    break
                j += 1
            if improved:
                continue
            else:
                i += 1
        if d_pert < best_dist:
            best_dist = d_pert
            best_tour = perturbed.copy()
            report_best_tour(best_tour)
    
    return best_tour