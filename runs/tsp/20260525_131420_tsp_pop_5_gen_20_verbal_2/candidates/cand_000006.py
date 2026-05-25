import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    if budget < n:
        tour = np.arange(n)
        rng.shuffle(tour)
        report_best_tour(tour)
        return tour
    # Regret insertion construction
    start = rng.randint(n)
    dists = distance_matrix[start]
    second = np.argmin(dists)
    if second == start:
        second = (start + 1) % n
    tour = [start, second]
    remaining = set(range(n)) - {start, second}
    while remaining:
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
    tour = np.array(tour, dtype=np.int32)
    best_tour = tour.copy()
    best_dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    # 2-opt improvement
    construction_steps = n - 2
    remaining_budget = budget - construction_steps
    if remaining_budget > 0:
        improved = True
        iteration = 0
        while improved and iteration < remaining_budget:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    iteration += 1
                    if iteration >= remaining_budget:
                        break
                    a, b, c, d = best_tour[i], best_tour[i+1], best_tour[j], best_tour[(j+1)%n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                        best_dist = best_dist + delta
                        report_best_tour(best_tour)
                        improved = True
                        break
                if improved or iteration >= remaining_budget:
                    break
    return best_tour