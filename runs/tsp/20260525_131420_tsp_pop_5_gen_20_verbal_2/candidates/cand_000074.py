import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)

    k = min(n - 1, 50)  # expanded candidate list
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
    restart_threshold = max(10, budget // 10)

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
                    return best_tour
            if iteration >= budget:
                return best_tour

        # Or-opt local search (relocate segments of length 2)
        improved = True
        while improved and iteration < budget:
            improved = False
            for i in range(n):
                if iteration >= budget:
                    break
                # consider segment from i to i+L-1 (wrap around)
                L = 2  # segment length
                seg_start = i
                seg_end = (i + L - 1) % n
                # skip if segment wraps around start (too complicated)
                if seg_end < seg_start:
                    continue
                # extract segment
                seg = tour[seg_start:seg_end+1]
                # remove segment: tour becomes two parts
                tour_before = tour[:seg_start]
                tour_after = tour[seg_end+1:] if seg_end+1 <= n else []
                tour_no_seg = np.concatenate([tour_before, tour_after])
                # try inserting segment in all possible positions
                best_pos = -1
                best_delta = 0.0
                m = len(tour_no_seg)
                for k_pos in range(m):
                    city_before = tour_no_seg[k_pos % m]
                    city_after = tour_no_seg[(k_pos + 1) % m]
                    # cost of removing segment originally
                    orig_remove_cost = distance_matrix[tour[seg_start], tour[(seg_start-1)%n]] + distance_matrix[tour[seg_end], tour[(seg_end+1)%n]]
                    # cost of reconnecting at new position
                    new_connect_cost = distance_matrix[city_before, seg[0]] + distance_matrix[seg[-1], city_after]
                    # cost of old edge at new position
                    old_edge_cost = distance_matrix[city_before, city_after]
                    # cost of new edges inside segment remain same?
                    # Actually, the segment's internal edges stay same.
                    # delta = new_connect_cost + old_remove_cost? Wait, careful.
                    # We removed segment and then reinsert it.
                    # Original total distance includes: edges around segment and internal edges.
                    # New total distance includes: edges around new position and internal edges.
                    # So delta = (new_connect_cost - old_edge_cost) - (old_remove_cost - (distance between seg_start prev and seg_end next?))
                    # Better: compute directly:
                    # current_dist known.
                    # If we move segment from its original location to new location, we change edges:
                    # - Remove edges: (tour[(seg_start-1)%n], seg[0]), (seg[-1], tour[(seg_end+1)%n]), and (city_before, city_after)
                    # - Add edges: (tour[(seg_start-1)%n], tour[(seg_end+1)%n]), (city_before, seg[0]), (seg[-1], city_after)
                    # So delta = (distance_matrix[tour[(seg_start-1)%n], tour[(seg_end+1)%n]] + distance_matrix[city_before, seg[0]] + distance_matrix[seg[-1], city_after]) - (distance_matrix[tour[(seg_start-1)%n], seg[0]] + distance_matrix[seg[-1], tour[(seg_end+1)%n]] + distance_matrix[city_before, city_after])
                    # In general, if segment is at start or end, handle wrap.
                    prev_seg = tour[(seg_start-1)%n]
                    next_seg = tour[(seg_end+1)%n]
                    delta = (distance_matrix[prev_seg, next_seg] + distance_matrix[city_before, seg[0]] + distance_matrix[seg[-1], city_after]) - (distance_matrix[prev_seg, seg[0]] + distance_matrix[seg[-1], next_seg] + distance_matrix[city_before, city_after])
                    if delta < best_delta:  # negative improvement
                        best_delta = delta
                        best_pos = k_pos
                if best_delta < -1e-12:
                    # apply move
                    # rebuild tour with segment inserted at best_pos
                    new_tour = np.concatenate([tour_no_seg[:best_pos], seg, tour_no_seg[best_pos:]])
                    # update pos and distance
                    pos = build_pos(new_tour)
                    current_dist += best_delta
                    tour = new_tour
                    iteration += 1
                    if current_dist < best_dist - 1e-12:
                        best_dist = current_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                        last_improvement = iteration
                    improved = True
                if improved:
                    break
            if iteration >= budget:
                return best_tour

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
            iteration += 1
        else:
            iteration += 1  # to progress even if no restart
    return best_tour