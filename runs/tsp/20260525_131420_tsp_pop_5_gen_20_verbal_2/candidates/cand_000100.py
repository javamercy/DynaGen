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
            candidates = []
            for city in unvisited:
                best_cost = float('inf')
                best_pos = -1
                second_best = float('inf')
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i + 1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < best_cost:
                        second_best = best_cost
                        best_cost = cost
                        best_pos = i + 1
                    elif cost < second_best:
                        second_best = cost
                regret = second_best - best_cost
                candidates.append((city, best_cost, best_pos, regret))
            max_regret = max(c[3] for c in candidates)
            chosen = [c for c in candidates if c[3] == max_regret]
            city, _, pos, _ = chosen[np.random.randint(len(chosen))]
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
    restart_threshold = max(10, budget // 10)

    while iteration < budget:
        improved = False
        # 2-opt
        for i in range(n - 2):
            if iteration >= budget:
                break
            iteration += 1
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
                    if current_dist < best_dist - 1e-12:
                        best_dist = current_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                        last_improvement = iteration
                    improved = True
                    break
            if improved:
                break
        if iteration >= budget:
            break
        
        # limited 3-opt: move segment of 2 consecutive cities
        if not improved and iteration < budget:
            for i in range(n):
                if iteration >= budget:
                    break
                iteration += 1
                # segment: cities at positions (i+1) and (i+2) mod n
                seg0 = tour[(i+1) % n]
                seg1 = tour[(i+2) % n]
                # remove segment
                a = tour[i]
                b = tour[(i+3) % n]
                rem_cost = distance_matrix[a, seg0] + distance_matrix[seg0, seg1] + distance_matrix[seg1, b]
                # try inserting after each j != i, i+1, i+2
                best_delta = 0
                best_j = -1
                for j in range(n):
                    if j == i or j == (i+1)%n or j == (i+2)%n:
                        continue
                    x = tour[j]
                    y = tour[(j+1)%n]
                    add_cost = distance_matrix[x, seg0] + distance_matrix[seg0, seg1] + distance_matrix[seg1, y]
                    delta = add_cost - rem_cost + distance_matrix[a, b]  # original connection after removal
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_j = j
                if best_delta < -1e-12:
                    # apply move: remove segment and insert after best_j
                    # construct new tour
                    indices = list(range(n))
                    seg_indices = [(i+1)%n, (i+2)%n]
                    remaining = [idx for idx in indices if idx not in seg_indices]
                    # reorder remaining: segment stays together and inserted after best_j
                    # simpler: delete and insert
                    new_tour = np.delete(tour, [(i+1)%n, (i+2)%n])
                    # adjust insert index
                    insert_pos = best_j + 1
                    if best_j >= i:
                        # account for removal of two earlier indices
                        if (i+1)%n > best_j:
                            insert_pos = best_j + 1
                        else:
                            insert_pos = best_j - 1  # because two removed before
                    else:
                        insert_pos = best_j + 1
                    new_tour = np.insert(new_tour, insert_pos, [seg0, seg1])
                    tour = new_tour
                    pos = build_pos(tour)
                    current_dist += best_delta
                    if current_dist < best_dist - 1e-12:
                        best_dist = current_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                        last_improvement = iteration
                    improved = True
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