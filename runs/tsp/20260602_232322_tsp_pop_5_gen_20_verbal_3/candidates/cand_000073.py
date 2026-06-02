import numpy as np
import random

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = float('inf')
    restarts = min(20, n)
    for _ in range(restarts):
        # Random initial triangle
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)
        # Regret insertion
        while remaining:
            best_city = -1
            best_regret = -1e9
            best_pos = -1
            for city in remaining:
                deltas = []
                for pos in range(len(tour)):
                    before = tour[pos - 1]
                    after = tour[pos] if pos < len(tour) else tour[0]
                    delta = distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
                    deltas.append((delta, pos))
                deltas.sort(key=lambda x: x[0])
                first = deltas[0][0]
                second = deltas[1][0] if len(deltas) > 1 else first
                regret = second - first
                if regret > best_regret or (abs(regret - best_regret) < 1e-12 and city < best_city):
                    best_regret = regret
                    best_city = city
                    best_pos = deltas[0][1]
            tour.insert(best_pos, best_city)
            remaining.remove(best_city)
        # Compute initial distance
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i + 1) % n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = np.array(tour, dtype=int)
            report_best_tour(best_tour)
        # VND: 2-opt and Or-opt
        improved = True
        while improved:
            improved = False
            # 2-opt first improvement
            for i in range(n - 1):
                for j in range(i + 2, n):
                    a = tour[i]
                    b = tour[(i + 1) % n]
                    c = tour[j]
                    d = tour[(j + 1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-10:
                        tour[i + 1:j + 1] = tour[j:i:-1]
                        dist += delta
                        if dist < best_dist - 1e-10:
                            best_dist = dist
                            best_tour = np.array(tour, dtype=int)
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Or-opt with incremental delta
            for L in [1, 2, 3]:
                if L >= n:
                    continue
                for i in range(n):
                    # Extract segment
                    if i + L <= n:
                        seg = tour[i:i + L]
                        rest = tour[:i] + tour[i + L:]
                        before = tour[(i - 1) % n]
                        after = tour[(i + L) % n]
                    else:
                        # wraparound
                        wrap_len = i + L - n
                        seg = tour[i:] + tour[:wrap_len]
                        rest = tour[wrap_len:i]
                        before = tour[(i - 1) % n]
                        after = tour[wrap_len]
                    m = len(rest)
                    if m == 0:
                        continue
                    removal_delta = distance_matrix[before, after] - (distance_matrix[before, seg[0]] + distance_matrix[seg[-1], after])
                    # Try insertions
                    for pos in range(m + 1):
                        if pos == 0:
                            c = rest[-1]
                            d = rest[0]
                        elif pos == m:
                            c = rest[-1]
                            d = rest[0]
                        else:
                            c = rest[pos - 1]
                            d = rest[pos]
                        insertion_delta = distance_matrix[c, seg[0]] + distance_matrix[seg[-1], d] - distance_matrix[c, d]
                        total_delta = removal_delta + insertion_delta
                        if total_delta < -1e-10:
                            # Apply move
                            new_tour = rest[:pos] + seg + rest[pos:]
                            tour = new_tour
                            dist += total_delta
                            if dist < best_dist - 1e-10:
                                best_dist = dist
                                best_tour = np.array(tour, dtype=int)
                                report_best_tour(best_tour)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return best_tour