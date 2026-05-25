import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    k = min(n - 1, 20)
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
            best = []
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
                        min_pos = i + 1
                    elif cost < second_min:
                        second_min = cost
                regret = second_min - min_cost
                best.append((city, min_cost, min_pos, regret))
            max_regret = max(c[3] for c in best)
            candidates = [c for c in best if c[3] == max_regret]
            city, _, pos, _ = candidates[np.random.randint(len(candidates))]
            tour.insert(pos, city)
            unvisited.remove(city)
        return np.array(tour, dtype=np.int32)

    tour = regret_insertion()
    pos = build_pos(tour)
    cur_dist = compute_dist(tour)
    best_tour = tour.copy()
    best_dist = cur_dist
    report_best_tour(best_tour)

    iteration = 0
    last_improvement = 0
    restart_threshold = max(10, budget // 10)

    while iteration < budget:
        improved = False
        for i in range(n - 2):
            if iteration >= budget:
                break
            a = tour[i]
            b = tour[i+1]
            for c in nn_list[b]:
                if iteration >= budget:
                    break
                iteration += 1
                j = pos[c]
                if j <= i+1 or j >= n-1:
                    continue
                d = tour[j+1]
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
        if not improved:
            if iteration - last_improvement > restart_threshold and cur_dist >= best_dist - 1e-12:
                tour = regret_insertion(start=np.random.randint(n))
                pos = build_pos(tour)
                cur_dist = compute_dist(tour)
                if cur_dist < best_dist - 1e-12:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                last_improvement = iteration
        # continue loop (iteration already incremented during evaluation)
    return best_tour