import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)

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

    def build_pos(tour):
        pos = np.empty(n, dtype=int)
        for idx, city in enumerate(tour):
            pos[city] = idx
        return pos

    def two_opt_steepest(tour, pos, dist):
        improved = True
        while improved:
            improved = False
            best_delta = 0.0
            best_i = -1
            best_j = -1
            for i in range(n-2):
                a = tour[i]
                b = tour[i+1]
                for j in range(i+2, n):
                    c = tour[j]
                    d = tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_delta < 0:
                i, j = best_i, best_j
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
                for idx in range(i+1, j+1):
                    pos[tour[idx]] = idx
                dist += best_delta
                improved = True
        return tour, pos, dist

    tour = regret_insertion()
    pos = build_pos(tour)
    dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    best_tour = tour.copy()
    best_dist = dist
    report_best_tour(best_tour)

    iteration = 0
    last_improvement = 0
    restart_threshold = max(10, budget // 15)

    while iteration < budget:
        # Steepest-ascent 2-opt until local optimum
        new_tour, new_pos, new_dist = two_opt_steepest(tour.copy(), pos.copy(), dist)
        iteration += 1
        if new_dist < best_dist - 1e-12:
            best_dist = new_dist
            best_tour = new_tour.copy()
            report_best_tour(best_tour)
            last_improvement = iteration
        tour, pos, dist = new_tour, new_pos, new_dist

        if iteration - last_improvement > restart_threshold and iteration < budget:
            # Perturbation: random 2-opt swap
            i = np.random.randint(0, n-2)
            j = np.random.randint(i+2, n)
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
            for idx in range(i+1, j+1):
                pos[tour[idx]] = idx
            a,b = tour[i], tour[(i+1)%n]
            c,d = tour[j], tour[(j+1)%n]
            delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
            dist += delta
            if dist < best_dist - 1e-12:
                best_dist = dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                last_improvement = iteration

            if iteration - last_improvement > restart_threshold:
                # Restart
                tour = regret_insertion()
                pos = build_pos(tour)
                dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
                if dist < best_dist - 1e-12:
                    best_dist = dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                last_improvement = iteration
            iteration += 1

    return best_tour