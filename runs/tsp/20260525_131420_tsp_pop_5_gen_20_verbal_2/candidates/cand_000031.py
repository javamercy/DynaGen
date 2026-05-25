import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)

    # candidate list size
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

    def nearest_neighbor(start=None):
        if start is None:
            start = np.random.randint(n)
        tour = [start]
        visited = {start}
        cur = start
        while len(tour) < n:
            # find nearest unvisited neighbor
            best = -1
            best_dist = float('inf')
            for j in range(n):
                if j not in visited:
                    d = distance_matrix[cur, j]
                    if d < best_dist:
                        best_dist = d
                        best = j
            tour.append(best)
            visited.add(best)
            cur = best
        return np.array(tour, dtype=np.int32)

    def double_bridge(tour):
        # standard double-bridge: split into 4 segments and reorder
        n = len(tour)
        # pick three distinct cut points
        a = np.random.randint(1, n//3)
        b = a + np.random.randint(1, n//3)
        c = b + np.random.randint(1, n//3)
        # segments: [0:a], [a:b], [b:c], [c:n]
        tour_new = np.concatenate([tour[0:a], tour[c:n], tour[b:c], tour[a:b]])
        return tour_new.astype(np.int32)

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
                        # apply move
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
    tour = nearest_neighbor()
    pos = build_pos(tour)
    best_tour = tour.copy()
    best_dist = compute_dist(tour)
    report_best_tour(best_tour)

    iteration = 0
    last_improvement = 0
    stagnation_threshold = max(10, budget // 10)
    perturbation_attempts = 3  # try perturbation a few times before full restart

    while iteration < budget:
        tour, cur_dist, iteration, last_improvement = two_opt(tour, pos, iteration, last_improvement)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        # stagnation check
        if iteration - last_improvement > stagnation_threshold and iteration < budget:
            # try double-bridge perturbation on current best
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
                # else run 2-opt on perturbed
                perturbed, d, iteration, _ = two_opt(perturbed, pos, iteration, last_improvement)
                if d < best_dist - 1e-12:
                    best_dist = d
                    best_tour = perturbed.copy()
                    report_best_tour(best_tour)
                    last_improvement = iteration
                    tour = perturbed
                    break
                # if perturbation didn't help, increment iteration
                iteration += 1
            # if still no improvement, do full restart with NN
            if iteration - last_improvement > stagnation_threshold and iteration < budget:
                tour = nearest_neighbor()
                pos = build_pos(tour)
                cur_dist = compute_dist(tour)
                if cur_dist < best_dist - 1e-12:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                last_improvement = iteration
    return best_tour