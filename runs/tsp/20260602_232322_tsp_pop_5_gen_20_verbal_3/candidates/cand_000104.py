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
        start = random.randrange(n)
        tour = [start]
        remaining = set(range(n)) - {start}
        while remaining:
            last = tour[-1]
            city = min(remaining, key=lambda c: distance_matrix[last, c])
            tour.append(city)
            remaining.remove(city)
        tour_arr = np.array(tour, dtype=np.int64)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour_arr[i], tour_arr[(i+1)%n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)
        improved = True
        while improved:
            improved = False
            # 2-opt
            for i in range(n):
                for j in range(i+2, n):
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
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
            if improved:
                continue
            # Or-opt L=2,3
            for L in [2,3]:
                if L >= n:
                    continue
                for i in range(n):
                    seg_indices = [(i+k)%n for k in range(L)]
                    seg = [tour[idx] for idx in seg_indices]
                    before = tour[(seg_indices[0]-1)%n]
                    after = tour[(seg_indices[-1]+1)%n]
                    removed_edges = distance_matrix[before, tour[seg_indices[0]]] + distance_matrix[tour[seg_indices[-1]], after]
                    added_edge = distance_matrix[before, after]
                    delta_rem = added_edge - removed_edges
                    new_tour_no_seg = [tour[j] for j in range(n) if j not in seg_indices]
                    m = n - L
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
                            break
                    if improved:
                        break
                if improved:
                    break
    return best_tour