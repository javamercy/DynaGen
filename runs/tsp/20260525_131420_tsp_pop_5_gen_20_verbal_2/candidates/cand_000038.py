import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour.copy())
        return tour
    rng = np.random.RandomState(seed)
    best_tour = None
    best_length = np.inf
    visited = set()
    for restart in range(5):
        start = rng.randint(n)
        dists = distance_matrix[start].copy()
        dists[start] = np.inf
        second = np.argmin(dists)
        tour = [start, second]
        remaining = set(range(n)) - {start, second}
        while remaining:
            best_regret = -1e100
            best_city = None
            best_pos = None
            best_cost = None
            for city in remaining:
                costs = []
                L = len(tour)
                for p in range(L):
                    left = tour[p]
                    right = tour[(p+1)%L]
                    cost = distance_matrix[left, city] + distance_matrix[city, right] - distance_matrix[left, right]
                    costs.append(cost)
                sorted_costs = sorted(costs)
                best = sorted_costs[0]
                second_best = sorted_costs[1] if len(sorted_costs) > 1 else best
                regret = second_best - best
                if regret > best_regret or (regret == best_regret and (best_cost is None or best < best_cost)):
                    best_regret = regret
                    best_city = city
                    best_pos = np.argmin(costs)
                    best_cost = best
            insert_idx = best_pos + 1
            tour = tour[:insert_idx] + [best_city] + tour[insert_idx:]
            remaining.remove(best_city)
        curr_tour = np.array(tour, dtype=np.int32)
        curr_length = sum(distance_matrix[curr_tour[i], curr_tour[(i+1)%n]] for i in range(n))
        if curr_length < best_length:
            best_length = curr_length
            best_tour = curr_tour.copy()
        visited.add(tuple(curr_tour))
    report_best_tour(best_tour.copy())
    current_tour = best_tour.copy()
    current_length = best_length
    attempts = 0
    improved = True
    while attempts < budget and improved:
        improved = False
        indices = list(range(n))
        rng.shuffle(indices)
        for i_idx in range(n-1):
            i = indices[i_idx]
            if attempts >= budget:
                break
            for j_idx in range(i_idx+2, n):
                j = indices[j_idx]
                if i > j:
                    i, j = j, i
                if j == i+1 or (i == 0 and j == n-1):
                    continue
                a = current_tour[i]
                b = current_tour[(i+1)%n]
                c = current_tour[j]
                d = current_tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                attempts += 1
                if new < old:
                    current_tour[i+1:j+1] = current_tour[i+1:j+1][::-1]
                    current_length += new - old
                    visited.add(tuple(current_tour))
                    if current_length < best_length:
                        best_length = current_length
                        best_tour = current_tour.copy()
                    report_best_tour(best_tour.copy())
                    improved = True
                    break
            if improved:
                break
        if not improved and attempts < budget:
            # double-bridge perturbation
            cut1 = rng.randint(1, n//2)
            cut2 = rng.randint(cut1+1, 3*n//4)
            cut3 = rng.randint(cut2+1, n-1)
            if cut1 < cut2 < cut3:
                new_tour = np.concatenate([current_tour[:cut1], current_tour[cut3:], current_tour[cut2:cut3], current_tour[cut1:cut2]])
                if new_tour.shape[0] == n and tuple(new_tour) not in visited:
                    current_tour = new_tour
                    current_length = sum(distance_matrix[current_tour[i], current_tour[(i+1)%n]] for i in range(n))
                    visited.add(tuple(current_tour))
                    attempts += 1
                    improved = True
    return best_tour