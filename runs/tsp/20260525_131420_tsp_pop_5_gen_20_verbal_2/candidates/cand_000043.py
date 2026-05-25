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
    tour_list = [start, second]
    remaining = set(range(n)) - {start, second}
    while remaining:
        best_regret = -1e100
        best_city = None
        best_pos = None
        best_cost = None
        for city in remaining:
            costs = []
            L = len(tour_list)
            for p in range(L):
                left = tour_list[p]
                right = tour_list[(p + 1) % L]
                cost = distance_matrix[left, city] + distance_matrix[city, right] - distance_matrix[left, right]
                costs.append(cost)
            sorted_costs = sorted(costs)
            best = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs) > 1 else best
            regret = second_best - best
            if regret > best_regret or (regret == best_regret and (best_cost is None or best < best_cost)):
                best_regret = regret
                best_city = city
                best_pos = int(np.argmin(costs))
                best_cost = best
        insert_idx = best_pos + 1
        tour_list = tour_list[:insert_idx] + [best_city] + tour_list[insert_idx:]
        remaining.remove(best_city)
    best_tour = np.array(tour_list, dtype=np.int32)
    best_length = sum(distance_matrix[best_tour[i], best_tour[(i + 1) % n]] for i in range(n))
    report_best_tour(best_tour.copy())
    current_tour = best_tour.copy()
    current_length = best_length
    epsilon = 0.1
    attempts = 0
    while attempts < budget:
        i = rng.randint(n)
        j = rng.randint(n)
        if i > j:
            i, j = j, i
        if j - i <= 1 or (i == 0 and j == n - 1):
            continue
        a = current_tour[i]
        b = current_tour[(i + 1) % n]
        c = current_tour[j]
        d = current_tour[(j + 1) % n]
        old = distance_matrix[a, b] + distance_matrix[c, d]
        new = distance_matrix[a, c] + distance_matrix[b, d]
        if new < old or rng.random() < epsilon:
            # reverse segment i+1..j
            current_tour[i + 1:j + 1] = current_tour[i + 1:j + 1][::-1]
            current_length += new - old
            if current_length < best_length:
                best_length = current_length
                best_tour = current_tour.copy()
                report_best_tour(best_tour.copy())
            attempts += 1
        else:
            attempts += 1
    return best_tour