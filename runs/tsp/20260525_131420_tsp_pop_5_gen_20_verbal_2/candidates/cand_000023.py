import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
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
                right = tour_list[(p+1)%L]
                cost = distance_matrix[left, city] + distance_matrix[city, right] - distance_matrix[left, right]
                costs.append(cost)
            sorted_costs = sorted(costs)
            best = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs) > 1 else best
            regret = second_best - best
            if regret > best_regret or (regret == best_regret and best_cost is None or best < best_cost):
                best_regret = regret
                best_city = city
                best_pos = np.argmin(costs)
                best_cost = best
        insert_idx = best_pos + 1
        tour_list = tour_list[:insert_idx] + [best_city] + tour_list[insert_idx:]
        remaining.remove(best_city)
    tour = np.array(tour_list, dtype=np.int32)
    best_tour = tour.copy()
    best_length = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))

    # 2-opt improvement with random ordering and probabilistic acceptance
    attempts = 0
    # Use a separate current tour for exploration
    cur_tour = tour.copy()
    cur_length = best_length
    while attempts < budget:
        # Choose random i<j such that j > i+1 and not (i==0 and j==n-1)
        i = rng.randint(n)
        j = rng.randint(n)
        if i > j:
            i, j = j, i
        if j == i+1 or (i == 0 and j == n-1):
            continue  # skip adjacent or wrap-around
        a = cur_tour[i]
        b = cur_tour[(i+1)%n]
        c = cur_tour[j]
        d = cur_tour[(j+1)%n]
        old = distance_matrix[a,b] + distance_matrix[c,d]
        new = distance_matrix[a,c] + distance_matrix[b,d]
        delta = new - old
        if delta < 0 or rng.rand() < 0.05:
            # Apply reversal
            cur_tour[i+1:j+1] = cur_tour[i+1:j+1][::-1]
            cur_length += delta
            if cur_length < best_length - 1e-9:  # allow numerical tolerance
                best_length = cur_length
                best_tour = cur_tour.copy()

        attempts += 1
    return best_tour