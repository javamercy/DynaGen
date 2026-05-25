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
    # Find farthest pair
    max_dist = -1
    pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i][j] > max_dist:
                max_dist = distance_matrix[i][j]
                pair = (i, j)
    tour = list(pair)
    remaining = set(range(n)) - set(pair)
    steps_left = budget - n + 2  # reserve budget for local search later
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
    report_best_tour(tour)
    best_tour = tour[:]
    best_len = tour_length(tour, distance_matrix)
    # 2-opt local search
    improved = True
    while improved and budget > 0:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                if j == i+1:
                    continue
                if j == n-1 and i == 0:
                    continue
                # compute delta
                a = tour[i], tour[(i+1)%n]
                b = tour[j], tour[(j+1)%n]
                delta = (distance_matrix[a[0], b[0]] + distance_matrix[a[1], b[1]] -
                         distance_matrix[a[0], a[1]] - distance_matrix[b[0], b[1]])
                if delta < 0:
                    new_tour = tour[i+1:j+1][::-1]
                    tour = tour[:i+1] + new_tour + tour[j+1:]
                    budget -= 1
                    new_len = best_len + delta
                    if new_len < best_len:
                        best_len = new_len
                        best_tour = tour[:]
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved:
                break
            if budget <= 0:
                break
        if budget <= 0:
            break
    return np.array(best_tour)

def tour_length(tour, dist):
    return sum(dist[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))