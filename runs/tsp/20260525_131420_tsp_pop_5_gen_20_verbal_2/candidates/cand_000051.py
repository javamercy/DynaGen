import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)

    k = min(n-1, 20)
    nn_list = []
    for i in range(n):
        order = np.argsort(distance_matrix[i])
        nn_list.append(order[1:k+1] if n > 1 else [])

    def build_pos(tour):
        pos = np.empty(n, dtype=int)
        for idx, city in enumerate(tour):
            pos[city] = idx
        return pos

    def compute_dist(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

    def regret_insertion(start=None):
        if start is None:
            start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best_candidates = []
            for city in unvisited:
                min_cost = float('inf')
                min_pos = -1
                second_min = float('inf')
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i+1) % len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < min_cost:
                        second_min = min_cost
                        min_cost = cost
                        min_pos = i+1
                    elif cost < second_min:
                        second_min = cost
                regret = second_min - min_cost
                best_candidates.append((city, min_cost, min_pos, regret))
            max_regret = max(c[3] for c in best_candidates)
            candidates = [c for c in best_candidates if c[3] == max_regret]
            city, _, pos, _ = candidates[np.random.randint(len(candidates))]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)

    tour = regret_insertion()
    pos = build_pos(tour)
    best_tour = tour.copy()
    best_dist = compute_dist(tour)
    report_best_tour(best_tour)

    cur_tour = tour.copy()
    cur_dist = best_dist
    cur_pos = pos.copy()
    iteration = 0
    no_improve = 0
    restart_threshold = max(10, budget // 10)

    while iteration < budget:
        improved = False
        for i in range(n-2):
            a = cur_tour[i]
            b = cur_tour[i+1]
            for c in nn_list[b]:
                j = cur_pos[c]
                if j <= i+1 or j >= n-1:
                    continue
                d = cur_tour[(j+1) % n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-12:
                    # apply 2-opt move
                    cur_tour[i+1:j+1] = cur_tour[i+1:j+1][::-1]
                    for idx in range(i+1, j+1):
                        cur_pos[cur_tour[idx]] = idx
                    cur_dist += delta
                    iteration += 1
                    if iteration >= budget:
                        break
                    if cur_dist < best_dist - 1e-12:
                        best_dist = cur_dist
                        best_tour = cur_tour.copy()
                        report_best_tour(best_tour)
                        no_improve = 0
                    else:
                        no_improve += 1
                    improved = True
                    break
            if improved:
                break
            if iteration >= budget:
                break
        if iteration >= budget:
            break
        if not improved:
            no_improve += 1
            iteration += 1
            if no_improve >= restart_threshold and iteration < budget:
                # restart
                new_tour = regret_insertion(start=np.random.randint(n))
                new_dist = compute_dist(new_tour)
                if new_dist < best_dist - 1e-12:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    report_best_tour(best_tour)
                cur_tour = new_tour
                cur_dist = new_dist
                cur_pos = build_pos(cur_tour)
                no_improve = 0

    return best_tour.astype(np.int32)