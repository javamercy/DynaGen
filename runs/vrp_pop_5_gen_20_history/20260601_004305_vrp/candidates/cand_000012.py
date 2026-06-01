import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        for _ in range(truck_count - m):
            routes.append([0, 0])
        return routes
    
    # --- TSP tour using nearest neighbor ---
    tour = []
    visited = [False] * n
    visited[0] = True
    current = 0
    for _ in range(m):
        best = -1
        best_dist = math.inf
        for v in range(1, n):
            if not visited[v]:
                d = distance_matrix[current][v]
                if d < best_dist or (d == best_dist and v < best):
                    best_dist = d
                    best = v
        tour.append(best)
        visited[best] = True
        current = best
    
    # --- DP optimal split to minimize max route distance ---
    # seg_dist[l][r] = distance of route covering segment tour[l:r] (inclusive of endpoints and depots)
    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0][tour[l]]
        for r in range(l+1, m+1):
            if r > l+1:
                acc += distance_matrix[tour[r-2]][tour[r-1]]
            if r == l+1:
                route_dist = distance_matrix[0][tour[l]] + distance_matrix[tour[l]][0]
            else:
                route_dist = acc + distance_matrix[tour[r-1]][0]
            seg_dist[l][r] = route_dist
    
    dp = [[math.inf] * (truck_count + 1) for _ in range(m + 1)]
    choice = [[-1] * (truck_count + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    for i in range(1, m+1):
        for t in range(1, min(i, truck_count) + 1):
            best = math.inf
            best_j = -1
            for j in range(t-1, i):
                if dp[j][t-1] < math.inf:
                    cand = max(dp[j][t-1], seg_dist[j][i])
                    if cand < best or (cand == best and j < best_j):
                        best = cand
                        best_j = j
            dp[i][t] = best
            choice[i][t] = best_j
    
    # reconstruct routes
    routes = []
    i = m
    t = truck_count
    while t > 0:
        j = choice[i][t]
        seg = tour[j:i]
        routes.append([0] + seg + [0])
        i = j
        t -= 1
    routes.reverse()
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    # helper functions
    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    def compute_max():
        return max(route_dist(r) for r in routes)
    
    best_routes = [list(r) for r in routes]
    best_max = compute_max()
    # report initial best
    report_best_vrp(best_routes)
    
    # ---- improvement phase ----
    max_iter = n * n
    for _ in range(max_iter):
        improved = False
        dists = [route_dist(r) for r in routes]
        order = sorted(range(len(routes)), key=lambda i: (-dists[i], i))
        
        # 2-opt on each route (longest first)
        for idx in order:
            route = routes[idx]
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_local_dist = route_dist(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < best_local_dist:
                        best_local_dist = new_dist
                        best_route = new_route
            if best_local_dist < route_dist(route):
                routes[idx] = best_route
                new_max = compute_max()
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                improved = True
                break  # restart after improvement
        if improved:
            continue
        
        # inter-route relocation (longest first)
        for src_idx in order:
            src_route = routes[src_idx]
            if len(src_route) <= 2:
                continue
            for cust in src_route[1:-1]:
                # find best insertion
                best_dst_idx = -1
                best_pos = -1
                best_new_max = None
                for dst_idx in range(len(routes)):
                    if dst_idx == src_idx:
                        continue
                    dst_route = routes[dst_idx]
                    for pos in range(1, len(dst_route)):
                        # compute new distances without actually modifying yet
                        new_src = [x for x in src_route if x != cust]
                        new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        new_src_dist = route_dist(new_src)
                        new_dst_dist = route_dist(new_dst)
                        other_max = 0
                        for r in range(len(routes)):
                            if r == src_idx or r == dst_idx:
                                continue
                            d = route_dist(routes[r])
                            if d > other_max:
                                other_max = d
                        cand_max = max(new_src_dist, new_dst_dist, other_max)
                        if candidate < best_max:  # potential improvement
                            if best_new_max is None or cand_max < best_new_max:
                                best_new_max = cand_max
                                best_dst_idx = dst_idx
                                best_pos = pos
                if best_new_max is not None:
                    # apply move
                    routes[src_idx] = [x for x in routes[src_idx] if x != cust]
                    routes[best_dst_idx] = routes[best_dst_idx][:best_pos] + [cust] + routes[best_dst_idx][best_pos:]
                    best_max = best_new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # balancing: move customer from longest route (best improvement)
        longest_idx = max(range(len(routes)), key=lambda i: (route_dist(routes[i]), i))
        longest = routes[longest_idx]
        if len(longest) > 2:
            best_move = None
            best_new_max = math.inf
            for cust in longest[1:-1]:
                new_longest = [x for x in longest if x != cust]
                new_longest_dist = route_dist(new_longest)
                for dst_idx in range(len(routes)):
                    if dst_idx == longest_idx:
                        continue
                    dst_route = routes[dst_idx]
                    for pos in range(1, len(dst_route)):
                        new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        new_dst_dist = route_dist(new_dst)
                        other_max = 0
                        for r in range(len(routes)):
                            if r == longest_idx or r == dst_idx:
                                continue
                            d = route_dist(routes[r])
                            if d > other_max:
                                other_max = d
                        cand_max = max(new_longest_dist, new_dst_dist, other_max)
                        if cand_max < best_new_max:
                            best_new_max = cand_max
                            best_move = (cust, dst_idx, pos)
            if best_move is not None and best_new_max < best_max:
                cust, dst_idx, pos = best_move
                routes[longest_idx] = [x for x in longest if x != cust]
                routes[dst_idx] = routes[dst_idx][:pos] + [cust] + routes[dst_idx][pos:]
                best_max = best_new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
                improved = True
        
        if not improved:
            break
    
    return best_routes