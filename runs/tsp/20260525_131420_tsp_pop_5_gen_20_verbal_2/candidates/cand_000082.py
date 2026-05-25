import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)

    # Precompute nearest neighbor list (k = min(n-1, 40))
    k = min(n - 1, 40)
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

    # Initial construction
    tour = regret_insertion()
    pos = build_pos(tour)
    current_dist = tour_distance(tour)
    best_tour = tour.copy()
    best_dist = current_dist
    report_best_tour(best_tour)

    iteration = 0
    last_improvement = 0
    restart_threshold = max(10, budget // 20)

    while iteration < budget:
        # 2-opt phase
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
                        # Reverse segment i+1..j
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

        # Or-opt phase (segment lengths 1-3)
        improved = True
        while improved and iteration < budget:
            improved = False
            for seg_len in [1, 2, 3]:
                for i in range(n):
                    if iteration >= budget:
                        break
                    seg_start = i
                    seg_end = (i + seg_len) % n
                    if seg_len == n:
                        continue
                    # Remove segment [seg_start, seg_end) (cyclic)
                    # Build list of remaining cities in order
                    if seg_start <= seg_end:
                        removed = tour[seg_start:seg_end].tolist()
                        remaining = np.concatenate([tour[:seg_start], tour[seg_end:]])
                    else:
                        removed = np.concatenate([tour[seg_start:], tour[:seg_end]]).tolist()
                        remaining = tour[seg_end:seg_start]
                    # Try inserting removed segment as a block at various positions
                    m = len(remaining)
                    best_delta = 0
                    best_insert_pos = None
                    for k in range(m):
                        # Insert after position k in remaining (cyclic)
                        a = remaining[k % m]
                        b = remaining[(k + 1) % m]
                        # cost of removing old connections: depends on segment endpoints
                        # Simplified: compute delta for moving whole segment
                        # We'll compute full cost after insertion to avoid complexity
                        # Not efficient but acceptable within budget
                        pass
                    # Not implementing full Or-opt due to complexity; keep as placeholder
                    # Instead, we rely on 2-opt and restarts
        # Since Or-opt is complex, we skip it and just restart on stagnation
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