import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    best_tour = np.arange(n)
    best_dist = np.inf
    for restart in range(15):
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)
        while remaining:
            best_city = -1
            best_regret = -1.0
            best_pos = -1
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
            tour.insert(best_pos, best_city)
            remaining.remove(best_city)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1)%n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)
        improved = True
        while improved:
            improved = False
            for L in [1,2,3]:
                if L >= n:
                    continue
                start_i = 0
                while start_i < n:
                    seg_indices = [(start_i + k) % n for k in range(L)]
                    seg = [tour[idx] for idx in seg_indices]
                    before_rem = tour[(seg_indices[0]-1)%n]
                    after_rem = tour[(seg_indices[-1]+1)%n]
                    removed_edges = distance_matrix[before_rem, tour[seg_indices[0]]] + distance_matrix[tour[seg_indices[-1]], after_rem]
                    added_edge = distance_matrix[before_rem, after_rem]
                    delta_rem = added_edge - removed_edges
                    new_tour_no_seg = [tour[j] for j in range(n) if j not in seg_indices]
                    m = n - L
                    found = False
                    for pos in range(m+1):
                        if pos == 0:
                            pre = new_tour_no_seg[-1]
                            post = new_tour_no_seg[0]
                        elif pos == m:
                            pre = new_tour_no_seg[-1]
                            post = new_tour_no_seg[0]
                        else:
                            pre = new_tour_no_seg[pos-1]
                            post = new_tour_no_seg[pos]
                        old_edge = distance_matrix[pre, post]
                        new_edges = distance_matrix[pre, seg[0]] + distance_matrix[seg[-1], post]
                        delta_ins = new_edges - old_edge
                        total_delta = delta_rem + delta_ins
                        if total_delta < -1e-10:
                            candidate = new_tour_no_seg[:pos] + seg + new_tour_no_seg[pos:]
                            dist += total_delta
                            tour = candidate
                            if dist < best_dist - 1e-10:
                                best_dist = dist
                                best_tour = np.array(tour)
                                report_best_tour(best_tour)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                    start_i += 1
                if improved:
                    break
            if improved:
                continue
            for i in range(n):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        dist += delta
                        if dist < best_dist - 1e-10:
                            best_dist = dist
                            best_tour = np.array(tour)
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
        if n >= 4:
            i, j = sorted(random.sample(range(1, n), 2))
            tour[i:j] = reversed(tour[i:j])
            dist = sum(distance_matrix[tour[t], tour[(t+1)%n]] for t in range(n))
            if dist < best_dist - 1e-10:
                best_dist = dist
                best_tour = np.array(tour)
                report_best_tour(best_tour)
    return best_tour