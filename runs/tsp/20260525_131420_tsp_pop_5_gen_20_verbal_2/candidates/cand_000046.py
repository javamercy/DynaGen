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
    # find farthest pair
    max_dist = -1
    pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i][j] > max_dist:
                max_dist = distance_matrix[i][j]
                pair = (i, j)
    tour = list(pair)
    remaining = set(range(n)) - set(pair)
    steps_left = budget - (n - 2)  # each insertion step consumes 1
    # regret insertion
    while remaining and steps_left > 0:
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
            if regret > best_regret or (regret == best_regret and (best_cost is None or best < best_cost)):
                best_regret = regret
                best_city = city
                best_pos = costs.index(best)
                best_cost = best
        tour = tour[:best_pos] + [best_city] + tour[best_pos:]
        remaining.remove(best_city)
        steps_left -= 1
    if remaining:
        tour.extend(sorted(remaining))
    # call report on initial tour
    report_best_tour(tour)
    # 2-opt local search
    improved = True
    while improved and steps_left > 0:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if steps_left <= 0:
                    break
                steps_left -= 1
                # evaluate 2-opt move (i, i+1) and (j, j+1)
                a = tour[i]
                b = tour[(i + 1) % n]
                c = tour[j]
                d = tour[(j + 1) % n]
                delta = (distance_matrix[a][c] + distance_matrix[b][d] -
                         distance_matrix[a][b] - distance_matrix[c][d])
                if delta < -1e-12:
                    # reverse segment (i+1, j)
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    report_best_tour(tour)
                    improved = True
                    break
            if improved or steps_left <= 0:
                break
    return np.array(tour)