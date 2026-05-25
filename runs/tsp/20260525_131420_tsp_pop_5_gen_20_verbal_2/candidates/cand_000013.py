import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = list(range(n))
        report_best_tour(tour)
        return np.array(tour)
    rng = np.random.RandomState(seed)

    # --- Construction: regret insertion with farthest start ---
    max_dist = -1
    pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i][j] > max_dist:
                max_dist = distance_matrix[i][j]
                pair = (i, j)
    tour = list(pair)
    remaining = set(range(n)) - set(pair)
    steps_left = budget - n + 2  # as in parent

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
            best_cost_city = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs) > 1 else best_cost_city
            regret = second_best - best_cost_city
            if regret > best_regret or (regret == best_regret and best_cost_city < best_cost):
                best_regret = regret
                best_city = city
                best_pos = costs.index(best_cost_city)
                best_cost = best_cost_city
        tour = tour[:best_pos] + [best_city] + tour[best_pos:]
        remaining.remove(best_city)
        steps_left -= 1

    if remaining:
        tour.extend(sorted(remaining))

    # --- Helper functions ---
    def tour_distance(t):
        return sum(distance_matrix[t[i], t[(i+1)%n]] for i in range(n))

    best_tour = tour[:]
    best_dist = tour_distance(best_tour)
    report_best_tour(best_tour)

    # --- Local search: 2-opt ---
    improved = True
    while steps_left > 0 and improved:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                if steps_left <= 0:
                    break
                steps_left -= 1
                a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
                current = distance_matrix[a][b] + distance_matrix[c][d]
                new = distance_matrix[a][c] + distance_matrix[b][d]
                if new < current:
                    tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                    improved = True
                    new_dist = best_dist - current + new
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_tour = tour[:]
                        report_best_tour(best_tour)
                    break
            if steps_left <= 0:
                break

    return np.array(best_tour)