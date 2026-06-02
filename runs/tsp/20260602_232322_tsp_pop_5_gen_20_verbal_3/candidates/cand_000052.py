import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    best_tour = None
    best_dist = float('inf')

    # Single restart
    start = random.sample(range(n), 3)
    tour = list(start)
    remaining = set(range(n)) - set(start)

    def delta_insert(city, pos):
        before = tour[pos-1]
        after = tour[pos] if pos < len(tour) else tour[0]
        return distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]

    while remaining:
        best_city = -1
        best_regret = -1
        best_pos = -1
        best_cost = float('inf')
        for city in remaining:
            costs = []
            for pos in range(len(tour)):
                costs.append((delta_insert(city, pos), pos))
            costs.sort(key=lambda x: x[0])
            first = costs[0][0]
            second = costs[1][0] if len(costs) > 1 else first
            regret = second - first
            if regret > best_regret or (regret == best_regret and city < best_city):
                best_regret = regret
                best_city = city
                best_pos = costs[0][1]
                best_cost = first
        tour.insert(best_pos, best_city)
        remaining.remove(best_city)

    # Compute initial distance
    dist = 0
    for i in range(n):
        dist += distance_matrix[tour[i], tour[(i+1)%n]]
    if dist < best_dist - 1e-10:
        best_dist = dist
        best_tour = np.array(tour)
        report_best_tour(best_tour)

    # Or-opt with delta evaluation
    improved = True
    while improved:
        improved = False
        for L in [1, 2, 3]:
            if L >= n:
                continue
            for i in range(n):
                # segment indices from i to i+L-1 (circular)
                seg_indices = [(i + k) % n for k in range(L)]
                seg = [tour[idx] for idx in seg_indices]
                # remove segment to get remaining tour
                new_tour = [tour[j] for j in range(n) if j not in seg_indices]
                m = len(new_tour)
                # evaluate insertion at each position
                for pos in range(m + 1):
                    # compute delta quickly
                    # original edges involving segment: before first, after last
                    first_idx = seg_indices[0]
                    last_idx = seg_indices[-1]
                    before = tour[(first_idx - 1) % n] if first_idx != 0 else tour[-1]
                    after = tour[(last_idx + 1) % n] if last_idx != n-1 else tour[0]
                    # remove edges: (before, first) and (last, after)
                    # add new edges if pos>0 and pos<m
                    delta = -distance_matrix[before, seg[0]] - distance_matrix[seg[-1], after]
                    if pos == 0:
                        # insert at beginning: before becomes after? Actually insert before first element
                        new_before = new_tour[-1] if m > 0 else None
                        new_after = new_tour[0] if m > 0 else None
                        if m == 0:
                            delta += 0  # only one node, no edges
                        else:
                            delta += distance_matrix[seg[-1], new_after] + distance_matrix[new_before, seg[0]] - distance_matrix[new_before, new_after]
                    elif pos == m:
                        new_before = new_tour[-1]
                        new_after = new_tour[0]
                        delta += distance_matrix[seg[-1], new_after] + distance_matrix[new_before, seg[0]] - distance_matrix[new_before, new_after]
                    else:
                        left = new_tour[pos-1]
                        right = new_tour[pos]
                        delta += distance_matrix[seg[-1], right] + distance_matrix[left, seg[0]] - distance_matrix[left, right]
                    if delta < -1e-10:
                        # apply move
                        candidate = new_tour[:pos] + seg + new_tour[pos:]
                        dist_new = dist + delta
                        tour = candidate
                        dist = dist_new
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