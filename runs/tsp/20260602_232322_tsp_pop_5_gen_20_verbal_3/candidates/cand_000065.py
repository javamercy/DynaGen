import numpy as np
import random

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    # Multi-start regret insertion, best of 10
    best_tour = None
    best_dist = float('inf')
    for _ in range(min(10, n)):
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)
        while remaining:
            best_city = -1
            best_regret = -1e9
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
                if regret > best_regret + 1e-12:
                    best_regret = regret
                    best_city = city
                    best_pos = deltas[0][1]
                elif abs(regret - best_regret) < 1e-12 and city < best_city:
                    best_city = city
                    best_pos = deltas[0][1]
            tour.insert(best_pos, best_city)
            remaining.remove(best_city)
        tour_arr = np.array(tour, dtype=int)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour_arr[i], tour_arr[(i+1)%n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)

    # VND local search (2-opt, then Or-opt with L=1,2,3)
    def vnd(tour, dist):
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
                        new_tour = tour.copy()
                        new_tour[i+1:j+1] = tour[j:i:-1]
                        tour = new_tour
                        dist += delta
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Or-opt for L=1,2,3
            for L in [1, 2, 3]:
                if L >= n:
                    continue
                break_outer = False
                for i in range(n):
                    seg = []
                    for k in range(L):
                        seg.append(tour[(i+k)%n])
                    new_tour_list = []
                    for k in range(n):
                        idx = (i+L+k)%n
                        new_tour_list.append(tour[idx])
                    m = len(new_tour_list)
                    for pos in range(m+1):
                        cand = new_tour_list[:pos] + seg + new_tour_list[pos:]
                        cand_dist = 0.0
                        for k in range(n):
                            cand_dist += distance_matrix[cand[k], cand[(k+1)%n]]
                        if cand_dist < dist - 1e-10:
                            tour = np.array(cand, dtype=int)
                            dist = cand_dist
                            improved = True
                            break_outer = True
                            break
                    if break_outer:
                        break
                if break_outer:
                    break
        return tour, dist

    # Apply VND to initial best
    best_tour, best_dist = vnd(best_tour, best_dist)
    report_best_tour(best_tour)

    # ILS with double-bridge perturbation
    max_iter = max(20, n//5)
    for _ in range(max_iter):
        tour = best_tour.copy()
        indices = sorted(random.sample(range(n), 4))
        a, b, c, d = indices
        new_tour = np.concatenate([
            tour[:a],
            tour[c:d],
            tour[b:c],
            tour[a:b],
            tour[d:]
        ]).astype(int)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[new_tour[i], new_tour[(i+1)%n]]
        new_tour, new_dist = vnd(new_tour, dist)
        if new_dist < best_dist - 1e-10:
            best_dist = new_dist
            best_tour = new_tour.copy()
            report_best_tour(best_tour)

    return best_tour