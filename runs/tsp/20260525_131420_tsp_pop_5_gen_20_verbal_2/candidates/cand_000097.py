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

    def double_bridge(tour):
        # random indices a<b<c<d
        a = np.random.randint(0, n - 3)
        b = np.random.randint(a + 1, n - 2)
        c = np.random.randint(b + 1, n - 1)
        d = np.random.randint(c + 1, n)
        # segments: [a:b], [b:c], [c:d], [d:...] plus wrap
        # New tour: [0:a], [c:d], [b:c], [a:b], [d:]
        new_tour = np.concatenate([
            tour[:a],
            tour[c:d],
            tour[b:c],
            tour[a:b],
            tour[d:]
        ])
        # ensure all cities present (it should be)
        return new_tour

    tour = regret_insertion()
    pos = build_pos(tour)
    current_dist = tour_distance(tour)
    best_tour = tour.copy()
    best_dist = current_dist
    report_best_tour(best_tour)

    iteration = 0
    last_improvement = -1
    restart_threshold = max(10, budget // 8)

    while iteration < budget:
        # Steepest-ascent 2-opt pass (best improvement)
        best_delta = 0.0
        best_i = -1
        best_j = -1
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
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        if best_delta < -1e-12:
            i, j = best_i, best_j
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
            for idx in range(i+1, j+1):
                pos[tour[idx]] = idx
            current_dist += best_delta
            if current_dist < best_dist - 1e-12:
                best_dist = current_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                last_improvement = iteration
        else:
            # No improvement via 2-opt; apply double-bridge kick
            if iteration < budget:
                iteration += 1
                tour = double_bridge(tour)
                pos = build_pos(tour)
                current_dist = tour_distance(tour)
                if current_dist < best_dist - 1e-12:
                    best_dist = current_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                    last_improvement = iteration
        
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