import numpy as np
import time

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0, 1])

    # ---------- Regret-2 construction ----------
    max_dist = -1
    best_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i][j]
            if d > max_dist:
                max_dist = d
                best_pair = (i, j)
    tour = list(best_pair)
    unvisited = set(range(n)) - set(tour)
    while unvisited:
        best_regret = -1
        best_city = None
        best_pos = None
        for k in unvisited:
            costs = []
            L = len(tour)
            for pos in range(L):
                i = tour[pos]
                j = tour[(pos+1) % L]
                cost = distance_matrix[i][k] + distance_matrix[k][j] - distance_matrix[i][j]
                costs.append(cost)
            sorted_costs = sorted(costs)
            best_cost = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs) > 1 else float('inf')
            regret = second_best - best_cost
            if regret > best_regret:
                best_regret = regret
                best_city = k
                best_pos = costs.index(best_cost)
        tour.insert(best_pos+1, best_city)
        unvisited.remove(best_city)

    tour_arr = np.array(tour)

    def tour_dist(t):
        d = 0.0
        for idx in range(n):
            d += distance_matrix[t[idx]][t[(idx+1)%n]]
        return d

    best_tour = tour_arr.copy()
    best_dist = tour_dist(best_tour)
    report_best_tour(best_tour)

    # ---------- Iterated local search with 2-opt (first-improvement, random order) and double-bridge ----------
    start_time = time.time()
    time_limit = 55  # seconds

    def two_opt(t: np.ndarray, dist: float) -> tuple:
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            indices = list(range(n))
            pairs = []
            for i in range(n):
                for j in range(i+2, n):
                    pairs.append((i, j))
            np.random.shuffle(pairs)
            for i, j in pairs:
                if time.time() - start_time >= time_limit:
                    return t, dist
                a = t[i]
                b = t[(i+1) % n]
                c = t[j]
                d = t[(j+1) % n]
                old = distance_matrix[a][b] + distance_matrix[c][d]
                new = distance_matrix[a][c] + distance_matrix[b][d]
                if new < old - 1e-10:
                    # apply reversal
                    t = np.concatenate((t[:i+1], t[i+1:j+1][::-1], t[j+1:]))
                    new_dist = dist - old + new
                    if new_dist < best_dist - 1e-10:
                        best_tour[:] = t
                        best_dist = new_dist
                        report_best_tour(t)
                    dist = new_dist
                    improved = True
                    break
        return t, dist

    def double_bridge(t: np.ndarray) -> np.ndarray:
        if n < 8:
            return t
        p = np.random.randint(0, n, 4)
        p.sort()
        a, b, c, d = p[0], p[1], p[2], p[3]
        segments = [t[:a], t[a:b], t[b:c], t[c:d], t[d:]]
        t_new = np.concatenate([segments[0], segments[2], segments[1], segments[3], segments[4]])
        return t_new

    current_tour = best_tour.copy()
    current_dist = best_dist
    iteration = 0
    while (time.time() - start_time) < time_limit and iteration < 20:
        # Perturbation
        perturbed = double_bridge(current_tour)
        perturbed_dist = tour_dist(perturbed)
        # Local search
        improved_tour, improved_dist = two_opt(perturbed, perturbed_dist)
        if improved_dist < current_dist - 1e-10:
            current_tour = improved_tour
            current_dist = improved_dist
            if current_dist < best_dist - 1e-10:
                best_tour = current_tour.copy()
                best_dist = current_dist
                report_best_tour(best_tour)
        else:
            # Simulated annealing acceptance (optional, small probability)
            if np.random.rand() < 0.02:
                current_tour = improved_tour
                current_dist = improved_dist
        iteration += 1

    return best_tour