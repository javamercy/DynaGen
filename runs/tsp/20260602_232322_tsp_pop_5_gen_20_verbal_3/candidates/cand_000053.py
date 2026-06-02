import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # convert to int to avoid floating point issues
    dist_int = (distance_matrix * 1e6).astype(np.int64)  # scale if needed, assume reasonable magnitude
    best_tour = None
    best_dist = np.iinfo(np.int64).max
    for restart in range(min(5, n)):
        # farthest insertion construction
        start = random.randint(0, n-1)
        tour = [start]
        remaining = set(range(n)) - {start}
        # add second city: farthest
        farthest_city = max(remaining, key=lambda c: dist_int[start, c])
        tour.append(farthest_city)
        remaining.remove(farthest_city)
        # add rest by farthest insertion
        while remaining:
            # find city with maximum minimum insertion cost
            best_city = None
            best_cost = -1
            best_pos = -1
            for city in remaining:
                min_cost = np.iinfo(np.int64).max
                best_pos_city = -1
                for pos in range(len(tour)):
                    before = tour[pos-1]
                    after = tour[pos] if pos < len(tour) else tour[0]
                    cost = dist_int[before, city] + dist_int[city, after] - dist_int[before, after]
                    if cost < min_cost:
                        min_cost = cost
                        best_pos_city = pos
                if min_cost > best_cost:
                    best_cost = min_cost
                    best_city = city
                    best_pos = best_pos_city
            tour.insert(best_pos, best_city)
            remaining.remove(best_city)
        # compute distance
        dist = 0
        for i in range(n):
            dist += dist_int[tour[i], tour[(i+1)%n]]
        if dist < best_dist:
            best_dist = dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)
        # Or-opt improvement
        improved = True
        while improved:
            improved = False
            for L in [1, 2, 3]:
                if L >= n:
                    continue
                for i in range(n):
                    seg_indices = [(i + k) % n for k in range(L)]
                    seg = [tour[idx] for idx in seg_indices]
                    before_rem = tour[(seg_indices[0] - 1) % n]
                    after_rem = tour[(seg_indices[-1] + 1) % n]
                    removed_edges = dist_int[before_rem, tour[seg_indices[0]]] + dist_int[tour[seg_indices[-1]], after_rem]
                    added_edge = dist_int[before_rem, after_rem]
                    delta_rem = added_edge - removed_edges
                    new_tour_no_seg = [tour[j] for j in range(n) if j not in seg_indices]
                    m = n - L
                    for pos in range(m + 1):
                        if pos == 0:
                            pre = new_tour_no_seg[-1]
                            post = new_tour_no_seg[0]
                        elif pos == m:
                            pre = new_tour_no_seg[-1]
                            post = new_tour_no_seg[0]
                        else:
                            pre = new_tour_no_seg[pos-1]
                            post = new_tour_no_seg[pos]
                        old_edge = dist_int[pre, post]
                        new_edges = dist_int[pre, seg[0]] + dist_int[seg[-1], post]
                        delta_ins = new_edges - old_edge
                        total_delta = delta_rem + delta_ins
                        if total_delta < 0:
                            candidate = new_tour_no_seg[:pos] + seg + new_tour_no_seg[pos:]
                            dist += total_delta
                            tour = candidate
                            if dist < best_dist:
                                best_dist = dist
                                best_tour = np.array(tour)
                                report_best_tour(best_tour)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        # 2-opt improvement
        improved2 = True
        while improved2:
            improved2 = False
            for i in range(n):
                for j in range(i+2, n):
                    # consider reversing segment i+1..j
                    i_next = (i+1) % n
                    j_next = (j+1) % n
                    old_edges = dist_int[tour[i], tour[i_next]] + dist_int[tour[j], tour[j_next]]
                    new_edges = dist_int[tour[i], tour[j]] + dist_int[tour[i_next], tour[j_next]]
                    if new_edges < old_edges:
                        # apply 2-opt move: reverse segment i+1..j
                        tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                        dist += new_edges - old_edges
                        if dist < best_dist:
                            best_dist = dist
                            best_tour = np.array(tour)
                            report_best_tour(best_tour)
                        improved2 = True
                        break
                if improved2:
                    break
    return best_tour