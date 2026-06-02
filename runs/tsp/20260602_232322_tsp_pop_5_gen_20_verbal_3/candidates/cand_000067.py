import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)

    best_tour = None
    best_cost = np.inf
    rcl_size = max(2, int(np.sqrt(n)))

    def compute_cost(tour):
        cost = 0.0
        for k in range(n - 1):
            cost += distance_matrix[tour[k], tour[k + 1]]
        cost += distance_matrix[tour[-1], tour[0]]
        return cost

    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == i + 1:
                        continue
                    a, b, c, d = tour[i], tour[(i + 1) % n], tour[j], tour[(j + 1) % n]
                    old = distance_matrix[a, b] + distance_matrix[c, d]
                    new = distance_matrix[a, c] + distance_matrix[b, d]
                    if new < old:
                        tour[i + 1:j + 1] = np.flip(tour[i + 1:j + 1])
                        improved = True
                        break
                if improved:
                    break
        return tour

    def double_bridge(tour):
        # Split tour into 4 segments and reorder: A, D, C, B
        p = np.random.randint(1, n // 3)
        q = np.random.randint(p + 1, 2 * n // 3)
        r = np.random.randint(q + 1, n - 1)
        # Ensure segments are valid (size > 0)
        if p == 0 or q == p or r == q:
            return tour.copy()
        # Build new tour: indices [0:p], [r:], [q:r], [p:q]
        new_tour = np.concatenate([tour[:p], tour[r:], tour[q:r], tour[p:q]])
        return new_tour

    for run in range(5):  # number of restarts
        # Randomized greedy construction
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            dists = distance_matrix[current, list(unvisited)]
            sorted_indices = np.argsort(dists)
            sorted_cities = np.array(list(unvisited))[sorted_indices]
            k = min(rcl_size, len(sorted_cities))
            choice = np.random.randint(k)
            next_city = sorted_cities[choice]
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        tour = np.array(tour, dtype=np.int64)
        # 2-opt
        tour = two_opt(tour)
        cost = compute_cost(tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(best_tour)
        # ILS iterations
        for ils_iter in range(25):
            perturbed = double_bridge(tour)
            perturbed = two_opt(perturbed)
            cost_pert = compute_cost(perturbed)
            # Accept if better or with small probability to avoid stagnation
            if cost_pert < cost or np.random.rand() < 0.05:
                tour = perturbed
                cost = cost_pert
                if cost_pert < best_cost:
                    best_cost = cost_pert
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
    return best_tour