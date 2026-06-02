import numpy as np
import random

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = float('inf')
    restarts = min(10, n)
    for _ in range(restarts):
        seed = random.sample(range(n), 3)
        tour = list(seed)
        remaining = set(range(n)) - set(seed)
        while remaining:
            best_city = None
            best_regret = -1.0
            best_pos = -1
            for city in remaining:
                deltas = []
                for pos in range(len(tour)):
                    before = tour[pos-1]
                    after = tour[pos] if pos < len(tour) else tour[0]
                    delta = distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
                    deltas.append((delta, pos))
                deltas.sort(key=lambda x: x[0])
                first = deltas[0][0]
                second = deltas[1][0] if len(deltas) > 1 else first
                regret = second - first
                if regret > best_regret or (regret == best_regret and (best_city is None or city < best_city)):
                    best_regret = regret
                    best_city = city
                    best_pos = deltas[0][1]
            tour.insert(best_pos, best_city)
            remaining.remove(best_city)
        tour = np.array(tour, dtype=int)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1)%n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        improved = True
        while improved:
            improved = False
            # 2-opt first improvement
            for i in range(n-1):
                for j in range(i+2, n):
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        tour[i+1:j+1] = tour[j:i:-1]
                        dist += delta
                        if dist < best_dist - 1e-10:
                            best_dist = dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Or-opt for segment lengths 1, 2, 3
            for L in (1, 2, 3):
                if L >= n:
                    continue
                for i in range(n):
                    if i + L <= n:
                        seg = tour[i:i+L].copy()
                        rest = np.concatenate([tour[:i], tour[i+L:]])
                    else:
                        seg = np.concatenate([tour[i:], tour[:i+L-n]])
                        rest = tour[i+L-n:i].copy()
                    m = len(rest)
                    if m == 0:
                        continue
                    for pos in range(m+1):
                        cand = np.concatenate([rest[:pos], seg, rest[pos:]])
                        cand_dist = 0.0
                        for k in range(n):
                            cand_dist += distance_matrix[cand[k], cand[(k+1)%n]]
                        if cand_dist < dist - 1e-10:
                            tour = cand
                            dist = cand_dist
                            if dist < best_dist - 1e-10:
                                best_dist = dist
                                best_tour = tour.copy()
                                report_best_tour(best_tour)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return best_tour