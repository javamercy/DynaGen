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
    # Initial tour via regret insertion
    start = rng.randint(n)
    dists = distance_matrix[start]
    second = np.argmin(dists)
    if second == start:
        second = (start + 1) % n
    tour = [start, second]
    remaining = set(range(n)) - {start, second}
    steps_left = budget - n + 2
    while remaining:
        if steps_left <= 0:
            tour.extend(sorted(remaining))
            break
        best_regret = -1.0
        best_city = None
        best_pos = None
        best_cost = np.inf
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
    # 2-opt improvement
    if steps_left > 0:
        improved = True
        while improved and steps_left > 0:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if steps_left <= 0:
                        break
                    a, b = tour[i], tour[(i+1)%n]
                    c, d = tour[j], tour[(j+1)%n]
                    current = distance_matrix[a, b] + distance_matrix[c, d]
                    new = distance_matrix[a, c] + distance_matrix[b, d]
                    if new < current:
                        tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                        improved = True
                        steps_left -= 1
                        break
                if improved:
                    break
    report_best_tour(tour)
    return np.array(tour)