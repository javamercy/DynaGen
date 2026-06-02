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

    restart_count = max(1, int(np.log2(n)))
    for restart in range(restart_count):
        # random triangle
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)

        # regret insertion with random tie-break
        while remaining:
            best_candidates = []
            best_regret = -1
            for city in remaining:
                costs = []
                for pos in range(len(tour)):
                    before = tour[pos-1]
                    after = tour[pos] if pos < len(tour) else tour[0]
                    delta = dist[before, city] + dist[city, after] - dist[before, after]
                    costs.append((delta, pos))
                costs.sort(key=lambda x: x[0])
                first = costs[0][0]
                second = costs[1][0] if len(costs) > 1 else first
                regret = second - first
                if regret > best_regret:
                    best_regret = regret
                    best_candidates = [(city, costs[0][1], first)]
                elif regret == best_regret:
                    best_candidates.append((city, costs[0][1], first))
            # random selection among ties
            city, best_pos, _ = random.choice(best_candidates)
            tour.insert(best_pos, city)
            remaining.remove(city)

        # compute initial distance
        current_dist = 0.0
        for i in range(n):
            current_dist += dist[tour[i], tour[(i+1)%n]]

        if current_dist < best_dist - 1e-10:
            best_dist = current_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)

        # Or-opt local search with delta evaluation (L=1,2,3 if n>=L)
        improved = True
        max_iters = max(10, n // 5)
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            segment_lengths = [L for L in [1, 2, 3] if L < n]
            for L in segment_lengths:
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

    return best_tour