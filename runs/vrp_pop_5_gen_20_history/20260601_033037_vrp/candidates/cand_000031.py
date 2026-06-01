import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        maxd = max(route_distance(r) for r in routes)
        if maxd < best_max - 1e-12:
            best_max = maxd
            best_routes = [list(r) for r in routes]

    # Nearest neighbor TSP tour (deterministic)
    unvisited = set(customers)
    current = 0
    perm = []
    while unvisited:
        nearest = min(unvisited, key=lambda x: (distance_matrix[current, x], x))
        perm.append(nearest)
        current = nearest
        unvisited.remove(nearest)

    def compute_segment_distance(perm, l, r):
        # returns distance of route: 0 -> perm[l] -> ... -> perm[r] -> 0
        d = distance_matrix[0, perm[l]]
        for i in range(l, r):
            d += distance_matrix[perm[i], perm[i+1]]
        d += distance_matrix[perm[r], 0]
        return d

    # Precompute all segment distances (upper triangular)
    m = len(perm)
    seg_dist = [[0.0]*m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            seg_dist[i][j] = compute_segment_distance(perm, i, j)

    # Collect all unique distances for binary search
    all_distances = set()
    for i in range(m):
        for j in range(i, m):
            all_distances.add(seg_dist[i][j])
    all_distances = sorted(all_distances)

    def feasibility_check(maxd, return_partition=False):
        # DP: dp[i] = min trucks to cover first i customers (0-indexed), with segment distance <= maxd
        dp = [float('inf')] * (m + 1)
        prev = [-1] * (m + 1)
        dp[0] = 0
        for i in range(1, m+1):
            # j is start index of last segment (1-indexed)
            for j in range(1, i+1):
                if seg_dist[j-1][i-1] <= maxd + 1e-12:
                    if dp[j-1] + 1 < dp[i]:
                        dp[i] = dp[j-1] + 1
                        prev[i] = j-1
                    elif dp[j-1] + 1 == dp[i] and (prev[i] == -1 or j-1 < prev[i]):
                        # tie-breaking: smaller previous index
                        prev[i] = j-1
        feasible = dp[m] <= truck_count
        if not return_partition:
            return feasible
        if not feasible:
            return None
        # reconstruct partition
        routes = []
        i = m
        while i > 0:
            j = prev[i]
            segment = perm[j:i]
            route = [0] + segment + [0]
            routes.append(route)
            i = j
        routes.reverse()
        # add empty routes if needed
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def best_split_for_permutation(perm):
        # recompute segment distances for this permutation
        m = len(perm)
        seg = [[0.0]*m for _ in range(m)]
        for i in range(m):
            for j in range(i, m):
                d = distance_matrix[0, perm[i]]
                for k in range(i, j):
                    d += distance_matrix[perm[k], perm[k+1]]
                d += distance_matrix[perm[j], 0]
                seg[i][j] = d
        all_d = set()
        for i in range(m):
            for j in range(i, m):
                all_d.add(seg[i][j])
        all_d = sorted(all_d)
        # binary search
        lo, hi = 0, len(all_d)-1
        while lo < hi:
            mid = (lo + hi) // 2
            if feasibility_check_given_seg(seg, all_d[mid], m):
                hi = mid
            else:
                lo = mid + 1
        maxd = all_d[lo]
        routes = reconstruct_given_seg(seg, maxd, m, perm)
        return routes, maxd

    def feasibility_check_given_seg(seg, maxd, m):
        dp = [float('inf')] * (m + 1)
        dp[0] = 0
        for i in range(1, m+1):
            for j in range(1, i+1):
                if seg[j-1][i-1] <= maxd + 1e-12:
                    dp[i] = min(dp[i], dp[j-1] + 1)
        return dp[m] <= truck_count

    def reconstruct_given_seg(seg, maxd, m, perm):
        dp = [float('inf')] * (m + 1)
        prev = [-1] * (m + 1)
        dp[0] = 0
        for i in range(1, m+1):
            for j in range(1, i+1):
                if seg[j-1][i-1] <= maxd + 1e-12:
                    if dp[j-1] + 1 < dp[i]:
                        dp[i] = dp[j-1] + 1
                        prev[i] = j-1
        routes = []
        i = m
        while i > 0:
            j = prev[i]
            routes.append([0] + perm[j:i] + [0])
            i = j
        routes.reverse()
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # Get initial routes
    initial_routes, _ = best_split_for_permutation(perm)
    report_best_vrp(initial_routes)

    # Hill climbing on permutation
    max_iters = min(500, len(perm) * truck_count * 2)
    current_perm = perm[:]
    _, current_max = best_split_for_permutation(current_perm)

    for iteration in range(max_iters):
        improved = False
        # Interchange (swap) of two customers
        m = len(current_perm)
        for i in range(m):
            for j in range(i+1, m):
                new_perm = current_perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_routes, new_max = best_split_for_permutation(new_perm)
                if new_max < current_max - 1e-12:
                    current_perm = new_perm
                    current_max = new_max
                    report_best_vrp(new_routes)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # 2-opt (reverse segment)
        for i in range(m):
            for j in range(i+2, m):
                new_perm = current_perm[:i] + current_perm[i:j+1][::-1] + current_perm[j+1:]
                new_routes, new_max = best_split_for_permutation(new_perm)
                if new_max < current_max - 1e-12:
                    current_perm = new_perm
                    current_max = new_max
                    report_best_vrp(new_routes)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best_routes