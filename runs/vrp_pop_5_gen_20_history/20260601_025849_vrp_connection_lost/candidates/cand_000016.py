import numpy as np
from typing import List

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> List[List[int]]:
    n = distance_matrix.shape[0]
    # Step 1: Generate a giant tour using nearest neighbor (deterministic)
    visited = set()
    current = 0
    order = []
    for _ in range(n - 1):
        min_dist = float('inf')
        best = -1
        for v in range(1, n):
            if v not in visited:
                d = distance_matrix[current, v]
                if d < min_dist or (d == min_dist and v < best):
                    min_dist = d
                    best = v
        visited.add(best)
        order.append(best)
        current = best
    L = len(order)
    if L == 0:
        return [[0, 0] for _ in range(truck_count)]

    # Precompute segment costs
    # intra_prefix[i] = sum_{k=0}^{i-1} d(order[k], order[k+1])
    intra_prefix = [0.0] * (L)
    for i in range(L - 1):
        intra_prefix[i+1] = intra_prefix[i] + distance_matrix[order[i], order[i+1]]

    # segment_cost(i,j) for customers i..j inclusive (0-indexed)
    def segment_cost(i, j):
        if i > j:
            return 0.0  # empty segment (should not happen)
        first = distance_matrix[0, order[i]]
        intra = intra_prefix[j] - intra_prefix[i]
        last = distance_matrix[order[j], 0]
        return first + intra + last

    # DP: dp[i][k] = min max distance for first i customers (0..i-1) using k routes
    max_non_empty = min(truck_count, L)
    INF = float('inf')
    dp = [[INF] * (max_non_empty + 1) for _ in range(L + 1)]
    parent = [[-1] * (max_non_empty + 1) for _ in range(L + 1)]
    dp[0][0] = 0.0
    for i in range(1, L + 1):
        for k in range(1, max_non_empty + 1):
            # j is number of customers in first k-1 routes, must be at least k-1
            for j in range(k - 1, i):
                seg_cost = segment_cost(j, i - 1)
                cand = max(dp[j][k-1], seg_cost)
                if cand < dp[i][k]:
                    dp[i][k] = cand
                    parent[i][k] = j
    k_used = max_non_empty
    i = L
    route_custs_list = []
    while k_used > 0:
        j = parent[i][k_used]
        # customers from j to i-1
        customers = order[j:i]
        route_custs_list.append(customers)
        i = j
        k_used -= 1
    route_custs_list.reverse()  # list of lists of customers per non-empty route
    routes = [[0] + custs + [0] for custs in route_custs_list]
    # Add empty routes
    while len(routes) < truck_count:
        routes.append([0, 0])

    def route_distance(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += distance_matrix[a, b]
        return d

    def max_route_distance(rts):
        return max(route_distance(r) for r in rts)

    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)

    # Improved: relocate from longest route
    max_iter = n * n
    for _ in range(max_iter):
        current_max = max_route_distance(routes)
        # Find longest route (first one with max distance, deterministic)
        longest_idx = -1
        longest_dist = -1.0
        for idx, r in enumerate(routes):
            d = route_distance(r)
            if d > longest_dist:
                longest_dist = d
                longest_idx = idx
        if longest_dist <= 0 or len(routes[longest_idx]) <= 3:
            break
        improved = False
        route_long = routes[longest_idx]
        # Try each customer (excluding depots)
        for pos in range(1, len(route_long) - 1):
            cust = route_long[pos]
            for other_idx, other_route in enumerate(routes):
                if other_idx == longest_idx:
                    continue
                for other_pos in range(1, len(other_route)):
                    new_other = other_route[:other_pos] + [cust] + other_route[other_pos:]
                    new_self = route_long[:pos] + route_long[pos+1:]
                    new_routes = [list(r) for r in routes]
                    new_routes[longest_idx] = new_self
                    new_routes[other_idx] = new_other
                    new_max = max_route_distance(new_routes)
                    if new_max < best_max:
                        best_max = new_max
                        best_routes = [list(r) for r in new_routes]
                        routes = new_routes
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_routes