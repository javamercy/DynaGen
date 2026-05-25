import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    # Handle small budget: if budget < n, just return random tour
    if budget < n:
        tour = list(range(n))
        rng.shuffle(tour)
        report_best_tour(tour)
        return np.array(tour)
    # Start with two cities: random start, nearest neighbor as second
    start = rng.randint(n)
    dists = distance_matrix[start]
    second = np.argmin(dists)
    if second == start:  # avoid self
        second = (start + 1) % n
    tour = [start, second]
    remaining = set(range(n)) - {start, second}
    # Track remaining budget for insertion steps (each insertion is one step)
    steps_left = budget - n + 2  # initial NN steps? Actually we already used 1 step for second? We'll count from now.
    # If budget is huge, this is positive. If not, we may fallback.
    # But we already handled budget < n, so here steps_left >= 0.
    while remaining:
        if steps_left <= 0:
            # insert remaining arbitrarily
            tour.extend(sorted(remaining))
            break
        best_regret = -1
        best_city = None
        best_pos = None
        best_cost = None
        for city in remaining:
            # compute insertion costs for all positions
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
        # Insert best city at best position
        tour = tour[:best_pos] + [best_city] + tour[best_pos:]
        remaining.remove(best_city)
        steps_left -= 1
    report_best_tour(tour)
    return np.array(tour)