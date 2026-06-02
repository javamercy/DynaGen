import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    dist = distance_matrix
    best_tour = None
    best_dist = float('inf')

    for restart in range(10):
        # random triangle
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)

        # regret insertion
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
                    delta = dist[before][city] + dist[city][after] - dist[before][after]
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

        # compute initial distance
        current_dist = 0.0
        for i in range(n):
            current_dist += dist[tour[i]][tour[(i+1)%n]]

        if current_dist < best_dist - 1e-10:
            best_dist = current_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)

        # VND: Or-opt (L=1,2) and 2-opt
        improved = True
        max_iters = 50
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Or-opt for L=1,2
            for L in [1, 2]:
                if L >= n:
                    continue
                for i in range(n):
                    seg_indices = [(i + k) % n for k in range(L)]
                    seg = [tour[idx] for idx in seg_indices]
                    remaining_tour = [tour[j] for j in range(n) if j not in seg_indices]
                    m = len(remaining_tour)
                    prev = tour[(i-1)%n]
                    nxt = tour[(i+L)%n]
                    orig_remove = dist[prev][tour[i]] + dist[tour[(i+L-1)%n]][nxt]
                    after_remove = dist[prev][nxt]
                    for pos in range(m + 1):
                        if pos == 0 or pos == m:
                            orig_replace = dist[remaining_tour[-1]][remaining_tour[0]]
                        else:
                            orig_replace = dist[remaining_tour[pos-1]][remaining_tour[pos]]
                        new_prev = remaining_tour[-1] if pos == 0 else remaining_tour[pos-1]
                        new_nxt = remaining_tour[0] if pos == m else remaining_tour[pos]
                        new_insert = dist[new_prev][seg[0]] + dist[seg[-1]][new_nxt]
                        delta = after_remove - orig_remove + new_insert - orig_replace
                        if delta < -1e-10:
                            new_tour = remaining_tour[:pos] + seg + remaining_tour[pos:]
                            tour = new_tour
                            current_dist += delta
                            if current_dist < best_dist - 1e-10:
                                best_dist = current_dist
                                best_tour = np.array(tour)
                                report_best_tour(best_tour)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt
            for i in range(n):
                for j in range(i+2, n):
                    delta = (dist[tour[i]][tour[j]] + dist[tour[i+1]][tour[(j+1)%n]] -
                             dist[tour[i]][tour[i+1]] - dist[tour[j]][tour[(j+1)%n]])
                    if delta < -1e-10:
                        # reverse segment i+1 to j
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        current_dist += delta
                        if current_dist < best_dist - 1e-10:
                            best_dist = current_dist
                            best_tour = np.array(tour)
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break

    return best_tour