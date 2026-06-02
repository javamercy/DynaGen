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
        # random triangle
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)
        # regret insertion
        def delta(city, pos):
            before = tour[pos-1]
            after = tour[pos] if pos < len(tour) else tour[0]
            return distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
        while remaining:
            best_city = -1
            best_regret = -1
            best_pos = -1
            best_cost = float('inf')
            for city in remaining:
                costs = []
                for pos in range(len(tour)):
                    costs.append((delta(city, pos), pos))
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
        # initial distance
        tour_arr = np.array(tour)
        dist = 0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1)%n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)
        # Or-opt: move segments of length L=1,2,3
        improved = True
        while improved:
            improved = False
            for L in [1, 2, 3]:
                if L >= n:
                    continue
                for i in range(n):
                    # segment from i to i+L-1 (inclusive, wrapping)
                    seg_indices = [(i + k) % n for k in range(L)]
                    seg = [tour[idx] for idx in seg_indices]
                    # remove segment
                    new_tour = [tour[j] for j in range(n) if j not in seg_indices]
                    # try inserting segment at all positions (including ends)
                    for pos in range(len(new_tour) + 1):
                        candidate = new_tour[:pos] + seg + new_tour[pos:]
                        # compute distance
                        cand_dist = 0
                        for k in range(n):
                            cand_dist += distance_matrix[candidate[k], candidate[(k+1)%n]]
                        if cand_dist < dist - 1e-10:
                            tour = candidate
                            dist = cand_dist
                            if dist < best_dist - 1e-10:
                                best_dist = dist
                                best_tour = np.array(tour)
                                report_best_tour(best_tour)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return best_tour