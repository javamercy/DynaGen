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
    # Regret insertion with farthest-start
    max_dist = -1.0
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
        best_regret = -1.0
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
    # Initial best distance
    best_tour = list(tour)
    best_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    # Steepest-ascent 2-opt improvement
    improved = True
    while steps_left > 0 and improved:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if steps_left <= 0:
                    break
                # Evaluate 2-opt move
                a = tour[i]
                b = tour[(i+1)%n]
                c = tour[j]
                d = tour[(j+1)%n]
                delta = (distance_matrix[a,c] + distance_matrix[b,d] -
                         distance_matrix[a,b] - distance_matrix[c,d])
                steps_left -= 1
                if delta < 0:
                    # Perform swap
                    if j == n-1:
                        new_tour = tour[:i+1] + tour[i+1:j+1][::-1]
                    else:
                        new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                    tour = new_tour
                    new_dist = best_dist + delta
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_tour = list(tour)
                        report_best_tour(best_tour)
                    improved = True
                    break  # restart i loop after change
            if steps_left <= 0:
                break
    return np.array(best_tour)