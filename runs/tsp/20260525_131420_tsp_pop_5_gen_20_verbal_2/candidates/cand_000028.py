import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)

    # Precompute nearest neighbor lists for each city (size k = min(n-1, 50))
    k = min(n-1, 50)
    nn_list = []
    for i in range(n):
        order = np.argsort(distance_matrix[i])
        nn_list.append(order[1:k+1] if n > 1 else [])

    # Build mapping from city to current tour index (for fast lookup)
    def build_pos(tour):
        pos = np.empty(n, dtype=int)
        for idx, city in enumerate(tour):
            pos[city] = idx
        return pos

    def regret_insertion(start=None):
        if start is None:
            start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best_costs = []
            for city in unvisited:
                min_cost = float('inf')
                min_pos = -1
                second_min_cost = float('inf')
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i+1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < min_cost:
                        second_min_cost = min_cost
                        min_cost = cost
                        min_pos = i+1
                    elif cost < second_min_cost:
                        second_min_cost = cost
                regret = second_min_cost - min_cost
                best_costs.append((city, min_cost, min_pos, regret))
            # Choose city with max regret, break ties randomly
            max_regret = max(c[3] for c in best_costs)
            candidates = [c for c in best_costs if c[3] == max_regret]
            city, cost, pos, _ = candidates[np.random.randint(len(candidates))]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)

    def two_opt(tour, pos, iteration, last_improvement):
        n = len(tour)
        local_best = tour.copy()
        local_best_dist = compute_dist(tour)
        improved = True
        while improved and iteration < budget:
            improved = False
            for i in range(n-2):
                a = tour[i]
                b = tour[i+1]
                for c in nn_list[b]:
                    j = pos[c]
                    if j <= i+1 or j >= n-1:  # ensure j > i+1 and j < n-1 to avoid duplicate edge
                        continue
                    d = tour[(j+1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        # Apply move
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        # Update positions incrementally
                        for idx in range(i+1, j+1):
                            pos[tour[idx]] = idx
                        # Update distance
                        local_dist = compute_dist(tour)
                        if local_dist < local_best_dist - 1e-12:
                            local_best = tour.copy()
                            local_best_dist = local_dist
                            report_best_tour(local_best)
                            last_improvement = iteration
                        improved = True
                        break
                if improved:
                    break
                iteration += 1
                if iteration >= budget:
                    return local_best, local_best_dist, iteration, last_improvement
            if iteration >= budget:
                break
        return local_best, local_best_dist, iteration, last_improvement

    def compute_dist(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

    # Initial tour
    tour = regret_insertion()
    pos = build_pos(tour)
    best_tour = tour.copy()
    best_dist = compute_dist(tour)
    report_best_tour(best_tour)

    iteration = 0
    last_improvement = 0
    restart_threshold = max(10, budget // 10)

    while iteration < budget:
        tour, cur_dist, iteration, last_improvement = two_opt(tour, pos, iteration, last_improvement)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        # Check for restart
        if iteration - last_improvement > restart_threshold and iteration < budget:
            # Regret insertion with random start and random tie-break
            tour = regret_insertion(start=np.random.randint(n))
            pos = build_pos(tour)
            cur_dist = compute_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
            last_improvement = iteration
    return best_tour