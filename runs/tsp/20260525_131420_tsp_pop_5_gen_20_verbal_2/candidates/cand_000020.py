import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour
    rng = np.random.RandomState(seed)
    # Regret-insertion construction
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
    best_tour = np.array(tour, dtype=np.int32)
    best_length = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour.copy())
    # Precompute all valid (i,j) pairs for 2-opt
    pairs = []
    for i in range(n-1):
        for j in range(i+2, n):
            if i == 0 and j == n-1:
                continue
            pairs.append((i, j))
    rng.shuffle(pairs)
    attempts = 0
    for i, j in pairs:
        if attempts >= budget:
            break
        attempts += 1
        a = best_tour[i]
        b = best_tour[(i+1)%n]
        c = best_tour[j]
        d = best_tour[(j+1)%n]
        old = distance_matrix[a,b] + distance_matrix[c,d]
        new = distance_matrix[a,c] + distance_matrix[b,d]
        if new < old:
            best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
            best_length = best_length + (new - old)
            report_best_tour(best_tour.copy())
        else:
            # Accept worsening with small probability
            if rng.random() < 0.05:
                best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                best_length = best_length + (new - old)
    return best_tour