import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    customers = list(range(1, n))
    routes = [[0, c, 0] for c in customers]
    dists = [2 * distance_matrix[0, c] for c in customers]
    # Deterministic greedy merging
    while len(routes) > truck_count:
        remaining = len(routes)
        best_i = best_j = -1
        best_max = float('inf')
        best_dist = float('inf')
        for i in range(remaining):
            for j in range(i + 1, remaining):
                last_i = routes[i][-2]
                first_j = routes[j][1]
                new_dist = (dists[i] + dists[j] - distance_matrix[0, last_i] -
                            distance_matrix[0, first_j] + distance_matrix[last_i, first_j])
                other_max = max(dists[k] for k in range(remaining) if k not in (i, j))
                new_max = max(other_max, new_dist)
                if new_max < best_max or (new_max == best_max and new_dist < best_dist):
                    best_max = new_max
                    best_dist = new_dist
                    best_i, best_j = i, j
        i, j = best_i, best_j
        merged = routes[i][:-1] + routes[j][1:]
        routes[i] = merged
        dists[i] = best_dist
        del routes[j]
        del dists[j]
    # Fill empty routes if needed
    while len(routes) < truck_count:
        routes.append([0, 0])
        dists.append(0.0)
    # Lightweight local search
    def route_dist(route):
        return sum(distance_matrix[route[k], route[k+1]] for k in range(len(route)-1))
    improved = True
    max_iter = 2 * n
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        cust_order = list(range(1, n))
        random.shuffle(cust_order)
        for cust in cust_order:
            # find current route
            curr_idx = None
            for idx, r in enumerate(routes):
                if cust in r:
                    curr_idx = idx
                    break
            if curr_idx is None:
                continue
            curr_route = routes[curr_idx]
            for tgt_idx in range(len(routes)):
                if tgt_idx == curr_idx:
                    continue
                tgt_route = routes[tgt_idx]
                for pos in range(1, len(tgt_route)):
                    # try insertion at pos (after node at index pos-1)
                    new_tgt = tgt_route[:pos] + [cust] + tgt_route[pos:]
                    new_curr = [x for x in curr_route if x != cust]
                    if len(new_curr) == 1:
                        new_curr = [0, 0]
                    new_curr_dist = route_dist(new_curr)
                    new_tgt_dist = route_dist(new_tgt)
                    # compute new max
                    all_dists = []
                    for r_idx, r in enumerate(routes):
                        if r_idx == curr_idx:
                            all_dists.append(new_curr_dist)
                        elif r_idx == tgt_idx:
                            all_dists.append(new_tgt_dist)
                        else:
                            all_dists.append(dists[r_idx])
                    new_max_val = max(all_dists)
                    if new_max_val < max(dists):
                        routes[curr_idx] = new_curr
                        routes[tgt_idx] = new_tgt
                        dists[curr_idx] = new_curr_dist
                        dists[tgt_idx] = new_tgt_dist
                        improved = True
                        break
                if improved:
                    break
    report_best_vrp(routes)
    return routes