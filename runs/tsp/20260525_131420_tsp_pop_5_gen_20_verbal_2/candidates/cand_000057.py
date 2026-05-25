import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    if budget < n:
        tour = list(range(n))
        rng.shuffle(tour)
        report_best_tour(tour)
        return np.array(tour)
    # Farthest pair start
    max_dist = -1
    pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i][j] > max_dist:
                max_dist = distance_matrix[i][j]
                pair = (i, j)
    tour = list(pair)
    remaining = set(range(n)) - set(pair)
    steps_left = budget - n + 2
    while remaining:
        if steps_left <= 0:
            tour.extend(sorted(remaining))
            break
        best_regret = -1
        best_city = None
        best_pos = None
        best_cost = None
        for city in remaining:
            costs = []
            cur_len = len(tour)
            for p in range(cur_len + 1):
                left = tour[p-1] if p > 0 else tour[-1]
                right = tour[p] if p < cur_len else tour[0]
                cost = (distance_matrix[left, city] + distance_matrix[city, right] -
                        distance_matrix[left, right])
                costs.append(cost)
            sorted_costs = sorted(costs)
            best = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs) > 1 else best
            regret = second_best - best
            if regret > best_regret or (regret == best_regret and best < best_cost):
                best_regret = regret
                best_city = city
                best_pos = costs.index(best)
                best_cost = best
        tour = tour[:best_pos] + [best_city] + tour[best_pos:]
        remaining.remove(best_city)
        steps_left -= 1
    # 2-opt local search with remaining steps
    improved = True
    while steps_left > 0 and improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                if steps_left <= 0:
                    break
                steps_left -= 1
                # cost of current edges (i, i+1) and (j, j+1)
                i_next = (i + 1) % n
                j_next = (j + 1) % n
                delta = (distance_matrix[tour[i], tour[j]] +
                         distance_matrix[tour[i_next], tour[j_next]] -
                         distance_matrix[tour[i], tour[i_next]] -
                         distance_matrix[tour[j], tour[j_next]])
                if delta < -1e-12:
                    # apply 2-opt swap
                    tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                    improved = True
                    report_best_tour(tour)
            if steps_left <= 0:
                break
    report_best_tour(tour)
    return np.array(tour)