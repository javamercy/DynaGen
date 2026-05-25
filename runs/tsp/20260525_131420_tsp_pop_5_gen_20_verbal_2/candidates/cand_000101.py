import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)

    k = min(n-1, 40)
    # precompute nearest neighbor lists
    nn_lists = [np.argsort(distance_matrix[i])[1:k+1] for i in range(n)]

    def compute_tour_distance(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))

    def regret_insertion():
        # start with random city
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        # candidate list for each unvisited city: nearest neighbors in current tour?
        # Instead, use precomputed nn lists to find insertion cost
        while unvisited:
            best_city = None
            best_regret = -1
            best_pos = -1
            best_cost = float('inf')
            for city in unvisited:
                # find insertion cost in current tour
                # compute minimal insertion cost and second minimal
                costs = []
                for i in range(len(tour)):
                    j = (i+1) % len(tour)
                    cost = distance_matrix[tour[i], city] + distance_matrix[city, tour[j]] - distance_matrix[tour[i], tour[j]]
                    costs.append(cost)
                # get two smallest
                if len(costs) >= 2:
                    sorted_idx = np.argsort(costs)
                    min_cost = costs[sorted_idx[0]]
                    second_min_cost = costs[sorted_idx[1]]
                    regret = second_min_cost - min_cost
                else:
                    min_cost = costs[0]
                    regret = 0
                if regret > best_regret or (regret == best_regret and min_cost < best_cost):
                    best_regret = regret
                    best_min_cost = min_cost
                    best_cost = min_cost
                    best_city = city
                    best_pos = sorted_idx[0] if len(costs) >= 2 else 0
            # insert best city at best_pos
            tour.insert(best_pos+1, best_city)
            unvisited.remove(best_city)
        return np.array(tour, dtype=np.int32)

    def double_bridge(tour):
        n = len(tour)
        a = np.random.randint(1, n//3)
        b = a + np.random.randint(1, n//3)
        c = b + np.random.randint(1, n//3)
        tour_new = np.concatenate([tour[:a], tour[c:], tour[b:c], tour[a:b]])
        return tour_new.astype(np.int32)

    def two_opt(tour, pos, iteration, last_improvement):
        n = len(tour)
        best_tour = tour.copy()
        best_dist = compute_tour_distance(tour)
        improved = True
        while improved and iteration < budget:
            improved = False
            for i in range(n-2):
                a = tour[i]
                b = tour[i+1]
                for c in nn_lists[b]:
                    j = pos[c]
                    if j <= i+1 or j >= n-1:
                        continue
                    d = tour[(j+1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        for idx in range(i+1, j+1):
                            pos[tour[idx]] = idx
                        # update distance incrementally
                        # recompute full distance for simplicity (but slow) - better incremental?
                        # For compactness, recompute whole tour distance once
                        # but we can approximate delta: new_dist = old_dist + delta
                        # We'll update best_dist if improved
                        if delta < 0:
                            best_dist += delta
                            if best_dist < compute_tour_distance(best_tour):  # keep actual best
                                best_tour = tour.copy()
                                report_best_tour(best_tour)
                                last_improvement = iteration
                        improved = True
                        break
                if improved:
                    break
                iteration += 1
                if iteration >= budget:
                    return best_tour, best_dist, iteration, last_improvement
        return best_tour, best_dist, iteration, last_improvement

    # initial tour
    tour = regret_insertion()
    pos = np.empty(n, dtype=int)
    for idx, city in enumerate(tour):
        pos[city] = idx
    best_tour = tour.copy()
    best_dist = compute_tour_distance(tour)
    report_best_tour(best_tour)

    iteration = 0
    last_improvement = 0
    stagnation_threshold = max(20, budget // 5)
    perturbation_attempts = 3

    while iteration < budget:
        tour, cur_dist, iteration, last_improvement = two_opt(tour, pos, iteration, last_improvement)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        # stagnation check
        if iteration - last_improvement > stagnation_threshold and iteration < budget:
            perturbed = double_bridge(best_tour)
            for _ in range(perturbation_attempts):
                if iteration >= budget:
                    break
                pos = np.empty(n, dtype=int)
                for idx, city in enumerate(perturbed):
                    pos[city] = idx
                cur_dist = compute_tour_distance(perturbed)
                if cur_dist < best_dist - 1e-12:
                    best_dist = cur_dist
                    best_tour = perturbed.copy()
                    report_best_tour(best_tour)
                    last_improvement = iteration
                    tour = perturbed
                    break
                perturbed, d, iteration, _ = two_opt(perturbed, pos, iteration, last_improvement)
                if d < best_dist - 1e-12:
                    best_dist = d
                    best_tour = perturbed.copy()
                    report_best_tour(best_tour)
                    last_improvement = iteration
                    tour = perturbed
                    break
                iteration += 1
            # if still no improvement, restart with new regret tour
            if iteration - last_improvement > stagnation_threshold and iteration < budget:
                tour = regret_insertion()
                pos = np.empty(n, dtype=int)
                for idx, city in enumerate(tour):
                    pos[city] = idx
                cur_dist = compute_tour_distance(tour)
                if cur_dist < best_dist - 1e-12:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                last_improvement = iteration
    return best_tour