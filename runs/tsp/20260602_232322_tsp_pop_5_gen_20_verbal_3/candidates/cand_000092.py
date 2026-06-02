import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = np.inf
    for _ in range(10):
        # farthest-insertion construction
        start = random.randint(0, n-1)
        tour = [start]
        remaining = set(range(n))
        remaining.remove(start)
        while remaining:
            # find farthest city from tour (minimum distance to any tour node)
            farthest_city = None
            max_min_dist = -1
            for city in remaining:
                min_dist = min(distance_matrix[city, t] for t in tour)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest_city = city
            # insert farthest_city at best position
            best_pos = None
            best_inc = np.inf
            for pos in range(len(tour)):
                before = tour[pos-1]
                after = tour[pos] if pos < len(tour) else tour[0]
                inc = distance_matrix[before, farthest_city] + distance_matrix[farthest_city, after] - distance_matrix[before, after]
                if inc < best_inc:
                    best_inc = inc
                    best_pos = pos
            tour.insert(best_pos, farthest_city)
            remaining.remove(farthest_city)
        tour_arr = np.array(tour, dtype=np.int64)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour_arr[i], tour_arr[(i+1)%n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)
        # VND
        improved = True
        while improved:
            improved = False
            # Or-opt (L=1,2,3)
            for L in [1,2,3]:
                if L >= n:
                    continue
                for start_i in range(n):
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
                if improved:
                    break
            if improved:
                continue
            # 2-opt
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
            if not improved:
                # perturbation: random 2-opt swap (non-improving)
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i > j:
                    i, j = j, i
                if j - i < 2 and not (i==0 and j==n-1):
                    # skip adjacent or same
                    continue
                if i==0 and j==n-1:
                    # skip
                    continue
                a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
                tour[i+1:j+1] = reversed(tour[i+1:j+1])
                dist += distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                # ignore update of best_dist; perturbation may worsen
    return best_tour