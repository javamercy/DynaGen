import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    # deterministic nearest neighbor giant tour
    unvisited = set(range(1, n))
    current = 0
    perm = []
    while unvisited:
        next_cust = min(unvisited, key=lambda c: (distance_matrix[current, c], c))
        perm.append(next_cust)
        unvisited.remove(next_cust)
        current = next_cust
    m = len(perm)
    
    # precompute segment distances (inclusive indices)
    seg_dist = np.zeros((m, m))
    for i in range(m):
        d = distance_matrix[0, perm[i]]
        for j in range(i, m):
            if j > i:
                d += distance_matrix[perm[j-1], perm[j]]
            seg_dist[i][j] = d + distance_matrix[perm[j], 0]
    
    INF = float('inf')
    # dp[i][k] = min max for first i customers (i from 0..m) with k routes
    dp = [[INF] * (truck_count + 1) for _ in range(m + 1)]
    pre = [[-1] * (truck_count + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for i in range(1, m + 1):
        max_k = min(truck_count, i)
        for k in range(1, max_k + 1):
            best_val = INF
            best_j = -1
            for j in range(k - 1, i):
                if dp[j][k-1] == INF:
                    continue
                cand = max(dp[j][k-1], seg_dist[j][i-1])
                if cand < best_val:
                    best_val = cand
                    best_j = j
            dp[i][k] = best_val
            pre[i][k] = best_j
    
    # choose best k <= truck_count with finite dp
    best_k = None
    best_val = INF
    for k in range(1, min(truck_count, m) + 1):
        if dp[m][k] < best_val:
            best_val = dp[m][k]
            best_k = k
    if best_k is None:
        best_k = 1
    
    # reconstruct routes
    routes = []
    idx = m
    k = best_k
    while k > 0:
        j = pre[idx][k]
        segment = perm[j:idx]
        route = [0] + segment + [0]
        routes.append(route)
        idx = j
        k -= 1
    routes.reverse()
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_dist():
        return max(route_dist(r) for r in routes)
    
    current_max = max_dist()
    report_best_vrp(routes)
    
    # Local search
    improved = True
    max_iter = n * truck_count
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            best_new_route = None
            best_new_dist = route_dist(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                    if new_dist < best_new_dist - 1e-12:
                        best_new_dist = new_dist
                        best_new_route = new_route
            if best_new_route is not None:
                old_dist = route_dist(route)
                if best_new_dist < old_dist - 1e-12:
                    routes[r_idx] = best_new_route
                    new_max = max_dist()
                    if new_max < current_max - 1e-12:
                        current_max = new_max
                        report_best_vrp(routes)
                    improved = True
                    break
        if improved:
            continue
        # Inter-route relocate
        for cust in range(1, n):
            # find current route of cust
            src_idx = None
            src_pos = None
            for idx, route in enumerate(routes):
                if cust in route:
                    src_idx = idx
                    src_pos = route.index(cust)
                    break
            if src_idx is None:
                continue
            src_route = routes[src_idx]
            if len(src_route) <= 2:
                continue
            new_src = src_route[:src_pos] + src_route[src_pos+1:]
            if len(new_src) == 2:
                new_src = [0, 0]
            new_src_dist = sum(distance_matrix[new_src[k], new_src[k+1]] for k in range(len(new_src)-1))
            best_tgt_idx = None
            best_new_tgt = None
            best_new_tgt_dist = None
            best_max_val = float('inf')
            for tgt_idx, tgt_route in enumerate(routes):
                if tgt_idx == src_idx:
                    continue
                if len(tgt_route) == 2:
                    new_tgt = [0, cust, 0]
                    new_tgt_dist = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    other_max = max(route_dist(routes[i]) for i in range(truck_count) if i != src_idx and i != tgt_idx)
                    cand_max = max(new_src_dist, new_tgt_dist, other_max)
                    if cand_max < best_max_val - 1e-12:
                        best_max_val = cand_max
                        best_tgt_idx = tgt_idx
                        best_new_tgt = new_tgt
                        best_new_tgt_dist = new_tgt_dist
                else:
                    for pos in range(1, len(tgt_route)):
                        new_tgt = tgt_route[:pos] + [cust] + tgt_route[pos:]
                        new_tgt_dist = sum(distance_matrix[new_tgt[k], new_tgt[k+1]] for k in range(len(new_tgt)-1))
                        other_max = max(route_dist(routes[i]) for i in range(truck_count) if i != src_idx and i != tgt_idx)
                        cand_max = max(new_src_dist, new_tgt_dist, other_max)
                        if cand_max < best_max_val - 1e-12:
                            best_max_val = cand_max
                            best_tgt_idx = tgt_idx
                            best_new_tgt = new_tgt
                            best_new_tgt_dist = new_tgt_dist
            if best_tgt_idx is not None and best_max_val < current_max - 1e-12:
                routes[src_idx] = new_src
                routes[best_tgt_idx] = best_new_tgt
                current_max = best_max_val
                report_best_vrp(routes)
                improved = True
                break
    return routes