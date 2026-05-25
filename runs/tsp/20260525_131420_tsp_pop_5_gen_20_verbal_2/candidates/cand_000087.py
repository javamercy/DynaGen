import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)

    # candidate list size
    k = min(n-1, 35)
    nn_list = []
    for i in range(n):
        order = np.argsort(distance_matrix[i])
        nn_list.append(order[1:k+1] if n > 1 else [])

    def build_pos(tour):
        pos = np.empty(n, dtype=int)
        for idx, city in enumerate(tour):
            pos[city] = idx
        return pos

    def regret_construction():
        # Start with a random city
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best_city = None
            best_cost = -np.inf
            best_insert_pos = None
            for city in unvisited:
                best_delta = float('inf')
                second_best_delta = float('inf')
                best_pos = -1
                for pos in range(len(tour)):
                    a = tour[pos]
                    b = tour[(pos+1) % len(tour)]
                    delta = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if delta < best_delta:
                        second_best_delta = best_delta
                        best_delta = delta
                        best_pos = pos
                    elif delta < second_best_delta:
                        second_best_delta = delta
                regret = second_best_delta - best_delta
                if regret > best_cost:
                    best_cost = regret
                    best_city = city
                    best_insert_pos = best_pos
            # insert between best_insert_pos and next
            tour.insert(best_insert_pos+1, best_city)
            unvisited.remove(best_city)
        return np.array(tour, dtype=np.int32)

    def double_bridge(tour):
        n = len(tour)
        a = np.random.randint(1, n//3)
        b = a + np.random.randint(1, n//3)
        c = b + np.random.randint(1, n//3)
        segments = [tour[:a], tour[a:b], tour[b:c], tour[c:]]
        new_tour = np.concatenate([segments[0], segments[3], segments[2], segments[1]])
        return new_tour.astype(np.int32)

    def compute_dist(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

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
                    if j <= i+1 or j >= n-1:
                        continue
                    d = tour[(j+1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        for idx in range(i+1, j+1):
                            pos[tour[idx]] = idx
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

    # initial tour
    tour = regret_construction()
    pos = build_pos(tour)
    best_tour = tour.copy()
    best_dist = compute_dist(tour)
    report_best_tour(best_tour)

    iteration = 0
    last_improvement = 0
    stagnation_threshold = max(10, budget // 10)
    perturbation_attempts = 3

    while iteration < budget:
        tour, cur_dist, iteration, last_improvement = two_opt(tour, pos, iteration, last_improvement)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        if iteration - last_improvement > stagnation_threshold and iteration < budget:
            for _ in range(perturbation_attempts):
                if iteration >= budget:
                    break
                perturbed = double_bridge(best_tour)
                pos = build_pos(perturbed)
                cur_dist = compute_dist(perturbed)
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
            if iteration - last_improvement > stagnation_threshold and iteration < budget:
                tour = regret_construction()  # full restart
                pos = build_pos(tour)
                cur_dist = compute_dist(tour)
                if cur_dist < best_dist - 1e-12:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                last_improvement = iteration
    return best_tour