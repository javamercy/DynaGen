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
        # Random triangle
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)
        # Regret insertion
        while remaining:
            best_city = -1
            best_regret = -1.0
            best_pos = -1
            deltas_list = []
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
                if regret > best_regret or (regret == best_regret and city < best_city):
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
        # VND with perturbation cycles
        for _ in range(3):
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
                # Or-opt with L=1,2,3
                for L in [1, 2, 3]:
                    if L >= n:
                        continue
                    for i in range(n):
                        if i + L <= n:
                            seg = list(tour[i:i+L])
                            new_tour_list = list(tour[:i]) + list(tour[i+L:])
                        else:
                            seg = list(tour[i:]) + list(tour[:i+L-n])
                            new_tour_list = list(tour[i+L-n:i])
                        m = len(new_tour_list)
                        if m == 0:
                            continue
                        for pos in range(m + 1):
                            cand = new_tour_list[:pos] + seg + new_tour_list[pos:]
                            cand_dist = 0.0
                            for k in range(n):
                                cand_dist += distance_matrix[cand[k], cand[(k+1)%n]]
                            if cand_dist < dist - 1e-10:
                                tour = np.array(cand, dtype=int)
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
            if not improved:
                # Perturbation: double-bridge
                if n < 8:
                    break
                i = random.randint(1, n-2)
                j = random.randint(i+1, n-1)
                k = random.randint(j+1, n)
                if k >= n:
                    k = n-1
                if j >= k:
                    continue
                new_tour = np.concatenate([tour[:i], tour[k:], tour[i:j], tour[j:k]])
                tour = new_tour
                dist = 0.0
                for x in range(n):
                    dist += distance_matrix[tour[x], tour[(x+1)%n]]
                if dist < best_dist - 1e-10:
                    best_dist = dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
            else:
                break
    return best_tour