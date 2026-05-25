import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)

    k = min(n - 1, 50)
    nn_list = []
    for i in range(n):
        order = np.argsort(distance_matrix[i])
        nn_list.append(order[1:k+1] if n > 1 else [])

    def build_pos(tour):
        pos = np.empty(n, dtype=np.int32)
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
                    b = tour[(i + 1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < min_cost:
                        second_min_cost = min_cost
                        min_cost = cost
                        min_pos = i + 1
                    elif cost < second_min_cost:
                        second_min_cost = cost
                regret = second_min_cost - min_cost
                best_costs.append((city, min_cost, min_pos, regret))
            max_regret = max(c[3] for c in best_costs)
            candidates = [c for c in best_costs if c[3] == max_regret]
            city, _, pos, _ = candidates[np.random.randint(len(candidates))]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)

    def tour_distance(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

    tour = regret_insertion()
    pos = build_pos(tour)
    current_dist = tour_distance(tour)
    best_tour = tour.copy()
    best_dist = current_dist
    report_best_tour(best_tour)

    iteration = 0
    last_improvement = 0
    restart_threshold = max(10, budget // 15)
    or_opt_threshold = 5

    while iteration < budget:
        # 2-opt local search
        improved = True
        while improved and iteration < budget:
            improved = False
            for i in range(n - 2):
                a = tour[i]
                b = tour[i + 1]
                for c in nn_list[b]:
                    j = pos[c]
                    if j <= i + 1 or j >= n - 1:
                        continue
                    d = tour[(j + 1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        for idx in range(i+1, j+1):
                            pos[tour[idx]] = idx
                        current_dist += delta
                        iteration += 1
                        if current_dist < best_dist - 1e-12:
                            best_dist = current_dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                            last_improvement = iteration
                        improved = True
                        break
                if improved:
                    break
                iteration += 1
                if iteration >= budget:
                    break
            if iteration >= budget:
                break
        if iteration >= budget:
            break

        # Or-opt local search (relocate 1 or 2 consecutive nodes)
        or_improved = True
        or_iter = 0
        while or_improved and or_iter < or_opt_threshold and iteration < budget:
            or_improved = False
            for seg_len in [1, 2]:
                for i in range(n):
                    j = (i + seg_len) % n
                    if j < i:  # wrap-around case, skip for simplicity
                        continue
                    # remove segment i..j (indices i to j inclusive, wrap not allowed)
                    if i + seg_len >= n:
                        continue
                    segment = tour[i:i+seg_len].copy()
                    remaining = np.concatenate([tour[:i], tour[i+seg_len:]])
                    # try inserting segment at each position in remaining
                    best_delta = 0
                    best_insert_pos = -1
                    for k in range(len(remaining)):
                        # delta when inserting segment before remaining[k]
                        # new edges: segment[0] connected to previous of k, segment[-1] to remaining[k]
                        prev = remaining[(k-1) % len(remaining)] if k > 0 else remaining[-1]
                        next_city = remaining[k]
                        old_edges = distance_matrix[prev, next_city]
                        new_edges = (0 if seg_len == 1 else distance_matrix[segment[0], segment[-1]]) + distance_matrix[prev, segment[0]] + distance_matrix[segment[-1], next_city]
                        # Actually carefully: when inserting segment, the removed edges are: between tour[i-1] and tour[i], and between tour[j] and tour[j+1]
                        # But since we are doing relocate (remove segment and insert elsewhere), delta computation is tricky.
                        # Instead we compute full distance change by comparing old and new tours.
                        # To avoid O(n) each time, we use incremental but simpler to just compute new tour and distance? Budget limited.
                        # For simplicity, we compute full distance of new tour and compare with current_dist.
                        new_tour = np.concatenate([remaining[:k], segment, remaining[k:]])
                        new_dist = sum(distance_matrix[new_tour[i], new_tour[(i+1)%n]] for i in range(n))
                        delta = new_dist - current_dist
                        if delta < best_delta - 1e-12:
                            best_delta = delta
                            best_insert_pos = k
                    if best_delta < -1e-12:
                        # apply move
                        new_tour = np.concatenate([remaining[:best_insert_pos], segment, remaining[best_insert_pos:]])
                        tour = new_tour
                        pos = build_pos(tour)
                        current_dist += best_delta
                        iteration += 1
                        if current_dist < best_dist - 1e-12:
                            best_dist = current_dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                            last_improvement = iteration
                        or_improved = True
                        break
                if or_improved:
                    break
            or_iter += 1
        if iteration >= budget:
            break

        # Restart if stagnant
        if iteration - last_improvement > restart_threshold and iteration < budget:
            new_start = np.random.randint(n)
            tour = regret_insertion(start=new_start)
            pos = build_pos(tour)
            current_dist = tour_distance(tour)
            if current_dist < best_dist - 1e-12:
                best_dist = current_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
            last_improvement = iteration
    return best_tour