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
    for restart in range(min(10, n)):
        # random triangle
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)
        # regret insertion
        while remaining:
            best_city = -1
            best_regret = -1
            best_pos = -1
            costs = []
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
        # initial tour
        tour = np.array(tour)
        dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        # Variable Neighborhood Descent: 2-opt and Or-opt
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
                    if distance_matrix[a,b] + distance_matrix[c,d] > distance_matrix[a,c] + distance_matrix[b,d]:
                        new_tour = tour.copy()
                        new_tour[i+1:j+1] = tour[j:i:-1] if i+1 <= j else tour[i+1:j+1][::-1]
                        # compute new distance efficiently
                        new_dist = dist - distance_matrix[a,b] - distance_matrix[c,d] + distance_matrix[a,c] + distance_matrix[b,d]
                        tour = new_tour
                        dist = new_dist
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
            # Or-opt (segment lengths 1..3)
            for L in [1, 2, 3]:
                if L >= n:
                    continue
                for i in range(n):
                    seg_indices = [(i + k) % n for k in range(L)]
                    seg = [tour[idx] for idx in seg_indices]
                    new_tour_list = [tour[j] for j in range(n) if j not in seg_indices]
                    for pos in range(len(new_tour_list) + 1):
                        cand = new_tour_list[:pos] + seg + new_tour_list[pos:]
                        cand_dist = sum(distance_matrix[cand[k], cand[(k+1)%n]] for k in range(n))
                        if cand_dist < dist - 1e-10:
                            tour = np.array(cand)
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