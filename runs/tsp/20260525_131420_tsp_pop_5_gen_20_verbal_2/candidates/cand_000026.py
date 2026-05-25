import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour
    rng = np.random.RandomState(seed)
    if budget < n:
        tour = np.arange(n, dtype=np.int32)
        rng.shuffle(tour)
        report_best_tour(tour)
        return tour
    # Precompute nearest neighbors (k = min(20, n-1))
    k = min(20, n-1)
    neighbors = np.zeros((n, k), dtype=np.int32)
    for i in range(n):
        order = np.argsort(distance_matrix[i])
        neighbors[i] = order[1:k+1]
    # Regret insertion construction
    start = rng.randint(n)
    dists = distance_matrix[start]
    second = np.argmin(dists)
    if second == start:
        second = (start + 1) % n
    tour = [start, second]
    remaining = set(range(n)) - {start, second}
    while remaining:
        best_regret = -1
        best_city = None
        best_pos = None
        best_cost = None
        for city in remaining:
            costs = []
            cur_len = len(tour)
            for p in range(cur_len + 1):
                left = tour[p-1] if p > 0 else tour[-1]
                right = tour[p] if p < cur_len else tour[0]
                cost = (distance_matrix[left, city] + distance_matrix[city, right] -
                        distance_matrix[left, right])
                costs.append(cost)
            sorted_costs = sorted(costs)
            best = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs) > 1 else best
            regret = second_best - best
            if regret > best_regret or (regret == best_regret and (best_cost is None or best < best_cost)):
                best_regret = regret
                best_city = city
                best_pos = costs.index(best)
                best_cost = best
        tour = tour[:best_pos] + [best_city] + tour[best_pos:]
        remaining.remove(best_city)
    tour = np.array(tour, dtype=np.int32)
    best_tour = tour.copy()
    best_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    construction_steps = n - 2
    remaining_budget = budget - construction_steps
    if remaining_budget <= 0:
        return best_tour
    def tour_length(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))
    def double_bridge(t):
        n_ = len(t)
        a = rng.randint(1, n_ // 3)
        b = rng.randint(a+1, n_ // 2)
        c = rng.randint(b+1, 2*n_ // 3)
        d = rng.randint(c+1, n_-1)
        if d >= n_:
            d = n_-1
        if a >= b or b >= c or c >= d:
            return t.copy()
        new_tour = np.concatenate([t[:a], t[c:d], t[b:c], t[a:b], t[d:]]).astype(np.int32)
        return new_tour
    attempts = 0
    no_improve_attempts = 0
    threshold = max(100, remaining_budget // 10)
    improved = True
    while attempts < remaining_budget:
        improved = False
        for i in range(n-2):
            if attempts >= remaining_budget:
                break
            city_i1 = tour[i+1]
            for idx in range(k):
                j = neighbors[city_i1, idx]
                if j <= i+1 or j >= n:
                    continue
                attempts += 1
                if attempts > remaining_budget:
                    break
                a_city = tour[i]
                b_city = tour[i+1]
                c_city = tour[j]
                d_city = tour[(j+1)%n]
                delta = (distance_matrix[a_city, c_city] + distance_matrix[b_city, d_city] -
                         distance_matrix[a_city, b_city] - distance_matrix[c_city, d_city])
                if delta < -1e-12 or (rng.rand() < 0.01 and delta < 1e-12):
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    tour_len = tour_length(tour)
                    if tour_len < best_dist - 1e-12:
                        best_dist = tour_len
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                        no_improve_attempts = 0
                    improved = True
                    break
            if improved or attempts >= remaining_budget:
                break
        if not improved:
            no_improve_attempts += 1
            if no_improve_attempts >= threshold:
                if attempts >= remaining_budget:
                    break
                tour = double_bridge(tour)
                attempts += 1
                no_improve_attempts = 0
        else:
            no_improve_attempts = 0
    return best_tour