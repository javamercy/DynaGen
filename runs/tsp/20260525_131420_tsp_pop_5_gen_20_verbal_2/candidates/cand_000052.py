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
    for restart in range(10):
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
                b = sorted_costs[0]
                second_best = sorted_costs[1] if len(sorted_costs) > 1 else b
                regret = second_best - b
                if regret > best_regret or (regret == best_regret and (best_cost is None or b < best_cost)):
                    best_regret = regret
                    best_city = city
                    best_pos = np.argmin(costs)
                    best_cost = b
            insert_idx = best_pos + 1
            tour = tour[:insert_idx] + [best_city] + tour[insert_idx:]
            remaining.remove(best_city)
        curr_tour = np.array(tour, dtype=np.int32)
        curr_length = sum(distance_matrix[curr_tour[i], curr_tour[(i+1)%n]] for i in range(n))
        if curr_length < best_length:
            best_length = curr_length
            best_tour = curr_tour.copy()
    report_best_tour(best_tour.copy())
    current_tour = best_tour.copy()
    current_length = best_length
    attempts = 0
    improved = True
    visited_tours = set()
    visited_tours.add(tuple(current_tour))
    while attempts < budget and improved:
        improved = False
        best_delta = 0
        best_i = best_j = None
        for i in range(n-1):
            if attempts >= budget:
                break
            for j in range(i+2, n):
                if attempts >= budget:
                    break
                if i == 0 and j == n-1:
                    continue
                a = current_tour[i]
                b = current_tour[(i+1)%n]
                c = current_tour[j]
                d = current_tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                attempts += 1
                if new < old:
                    delta = new - old
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        improved = True
            if attempts >= budget:
                break
        if improved:
            i, j = best_i, best_j
            current_tour[i+1:j+1] = current_tour[i+1:j+1][::-1]
            current_length += best_delta
            report_best_tour(current_tour.copy())
            if current_length < best_length:
                best_length = current_length
                best_tour = current_tour.copy()
            visited_tours.add(tuple(current_tour))
        else:
            if attempts < budget:
                for _ in range(5):
                    i = rng.randint(0, n-2)
                    j = rng.randint(i+2, n-1)
                    if i == 0 and j == n-1:
                        continue
                    new_tour = current_tour.copy()
                    new_tour[i+1:j+1] = new_tour[i+1:j+1][::-1]
                    if tuple(new_tour) not in visited_tours:
                        current_tour = new_tour
                        current_length = sum(distance_matrix[current_tour[k], current_tour[(k+1)%n]] for k in range(n))
                        visited_tours.add(tuple(current_tour))
                        attempts += 1
                        improved = True
                        break
                if not improved:
                    break
    return best_tour