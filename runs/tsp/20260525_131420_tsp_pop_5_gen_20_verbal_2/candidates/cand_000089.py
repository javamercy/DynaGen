import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)

    k = min(n-1, 80)
    nn_list = []
    for i in range(n):
        order = np.argsort(distance_matrix[i])
        nn_list.append(order[1:k+1] if n > 1 else [])

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
            max_regret = max(c[3] for c in best_costs)
            candidates = [c for c in best_costs if c[3] == max_regret]
            city, cost, pos, _ = candidates[np.random.randint(len(candidates))]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)

    def compute_dist(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

    tour = regret_insertion()
    pos = build_pos(tour)
    cur_dist = compute_dist(tour)
    best_tour = tour.copy()
    best_dist = cur_dist
    report_best_tour(best_tour)

    iteration = 0
    last_improvement = 0
    restart_threshold = max(10, budget // 25)

    def two_opt_improve():
        nonlocal iteration, cur_dist, best_dist, best_tour, last_improvement, tour, pos
        improved = False
        i = 0
        while i < n - 2 and iteration < budget:
            a = tour[i]
            b = tour[(i+1) % n]
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
                    cur_dist += delta
                    if cur_dist < best_dist - 1e-12:
                        best_dist = cur_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                        last_improvement = iteration
                    improved = True
                    break
            if improved:
                break
            i += 1
            iteration += 1
        return improved

    def three_opt_improve():
        nonlocal iteration, cur_dist, best_dist, best_tour, last_improvement, tour, pos
        improved = False
        for i in range(n-3):
            if iteration >= budget:
                break
            a = tour[i]
            b = tour[(i+1) % n]
            for j in range(i+2, n-1):
                if iteration >= budget:
                    break
                c = tour[j]
                d = tour[(j+1) % n]
                for k in range(j+2, n):
                    if iteration >= budget:
                        break
                    e = tour[k]
                    f = tour[(k+1) % n]
                    # consider two alternative: reverse segment (i+1..j) or (j+1..k) or both
                    # case 1: reverse (i+1..j)
                    delta1 = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta1 < -1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        for idx in range(i+1, j+1):
                            pos[tour[idx]] = idx
                        cur_dist += delta1
                        if cur_dist < best_dist - 1e-12:
                            best_dist = cur_dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                            last_improvement = iteration
                        improved = True
                        return improved
                    # case 2: reverse (j+1..k)
                    delta2 = distance_matrix[c, e] + distance_matrix[d, f] - distance_matrix[c, d] - distance_matrix[e, f]
                    if delta2 < -1e-12:
                        tour[j+1:k+1] = tour[j+1:k+1][::-1]
                        for idx in range(j+1, k+1):
                            pos[tour[idx]] = idx
                        cur_dist += delta2
                        if cur_dist < best_dist - 1e-12:
                            best_dist = cur_dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                            last_improvement = iteration
                        improved = True
                        return improved
                    # case 3: swap segments (i+1..j) and (j+1..k)
                    # This is more complex; skip for simplicity
                    iteration += 1
        return improved

    while iteration < budget:
        improved = two_opt_improve()
        if improved:
            continue
        # if 2-opt stuck, try limited 3-opt
        improved = three_opt_improve()
        if improved:
            continue
        # restart if stagnation
        if iteration - last_improvement > restart_threshold and iteration < budget:
            tour = regret_insertion(start=np.random.randint(n))
            pos = build_pos(tour)
            cur_dist = compute_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
            last_improvement = iteration
        else:
            # if no improvement and not restart, just increment iteration to eventually exit
            iteration += 1
    return best_tour