import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = float('inf')
    for restart in range(min(10, n)):
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)
        while remaining:
            best_city = -1
            best_regret = -1
            best_pos = -1
            best_cost = float('inf')
            for city in remaining:
                costs = []
                for pos in range(len(tour)):
                    before = tour[pos-1]
                    after = tour[pos] if pos < len(tour) else tour[0]
                    delta = distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
                    costs.append((delta, pos))
                costs.sort(key=lambda x: x[0])
                first = costs[0][0]
                second = costs[1][0] if len(costs) > 1 else first
                regret = second - first
                if regret > best_regret or (regret == best_regret and city < best_city):
                    best_regret = regret
                    best_city = city
                    best_pos = costs[0][1]
                    best_cost = first
            tour.insert(best_pos, best_city)
            remaining.remove(best_city)
        tour_arr = np.array(tour)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1)%n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)
        # Or-opt with delta evaluation
        improved = True
        while improved:
            improved = False
            for L in [1, 2, 3]:
                if L >= n:
                    continue
                for i in range(n):
                    # segment from i of length L (wrapping)
                    seg = [tour[(i+k)%n] for k in range(L)]
                    prev = tour[(i-1)%n]
                    first = seg[0]
                    last = seg[-1]
                    next_node = tour[(i+L)%n]
                    removal_delta = - (distance_matrix[prev, first] + distance_matrix[last, next_node]) + distance_matrix[prev, next_node]
                    # build remaining tour (without segment)
                    if i+L <= n:
                        new_tour = tour[:i] + tour[i+L:]
                    else:
                        new_tour = tour[i+L-n:] + tour[:i]
                    m = len(new_tour)
                    # try all insertion positions (edges to break)
                    for pos in range(m):
                        before = new_tour[pos-1] if pos > 0 else new_tour[-1]
                        after = new_tour[pos]
                        ins_delta = - distance_matrix[before, after] + distance_matrix[before, first] + distance_matrix[last, after]
                        total_delta = removal_delta + ins_delta
                        if total_delta < -1e-10:
                            # construct new full tour
                            new_full = new_tour[:pos] + seg + new_tour[pos:]
                            new_dist = dist + total_delta
                            if new_dist < best_dist - 1e-10:
                                best_dist = new_dist
                                best_tour = np.array(new_full)
                                report_best_tour(best_tour)
                            tour = new_full
                            dist = new_dist
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return best_tour