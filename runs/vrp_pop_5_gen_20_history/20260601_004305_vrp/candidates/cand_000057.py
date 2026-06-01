import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # --- Construction: TSP tour + DP minimax split ---
    tour = []
    visited = [False] * n
    visited[0] = True
    current = 0
    for _ in range(m):
        best = -1
        best_dist = math.inf
        for v in range(1, n):
            if not visited[v]:
                d = distance_matrix[current, v]
                if d < best_dist or (d == best_dist and v < best):
                    best_dist = d
                    best = v
        tour.append(best)
        visited[best] = True
        current = best
    
    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0, tour[l]]
        for r in range(l + 1, m + 1):
            if r > l + 1:
                acc += distance_matrix[tour[r - 2], tour[r - 1]]
            if r == l + 1:
                seg_dist[l][r] = distance_matrix[0, tour[l]] + distance_matrix[tour[l], 0]
            else:
                seg_dist[l][r] = acc + distance_matrix[tour[r - 1], 0]
    
    dp = [[math.inf] * (truck_count + 1) for _ in range(m + 1)]
    choice = [[-1] * (truck_count + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    for i in range(1, m + 1):
        for t in range(1, min(i, truck_count) + 1):
            best_val = math.inf
            best_j = -1
            for j in range(t - 1, i):
                if dp[j][t - 1] < math.inf:
                    cand = max(dp[j][t - 1], seg_dist[j][i])
                    if cand < best_val or (cand == best_val and j < best_j):
                        best_val = cand
                        best_j = j
            dp[i][t] = best_val
            choice[i][t] = best_j
    
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
    
    def route_dist(route):
        total = 0
        for a in range(len(route) - 1):
            total += distance_matrix[route[a], route[a+1]]
        return total
    
    def compute_max():
        maxd = 0
        for r in routes:
            d = route_dist(r)
            if d > maxd:
                maxd = d
        return maxd
    
    def copy_routes():
        return [list(r) for r in routes]
    
    best_max = compute_max()
    best_routes = copy_routes()
    report_best_vrp(best_routes)
    
    max_passes = 10 * n
    operators = ['2opt', 'relocate', 'swap']
    for _ in range(max_passes):
        improved = False
        # Determine longest route (ties: smallest index)
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(routes)), key=lambda i: (dists[i], -i))
        # Actually we need smallest index among those with max dist:
        max_dist = max(dists)
        longest_idx = min(i for i, d in enumerate(dists) if d == max_dist)
        
        for op in operators:
            if op == '2opt':
                # Intra-route 2-opt on longest route
                route = routes[longest_idx]
                if len(route) > 3:
                    best_dist_opt = route_dist(route)
                    best_route = route[:]
                    found = False
                    for i in range(1, len(route) - 2):
                        for j in range(i + 1, len(route) - 1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_dist = route_dist(new_route)
                            if new_dist < best_dist_opt - 1e-12:
                                best_dist_opt = new_dist
                                best_route = new_route
                                found = True
                            elif abs(new_dist - best_dist_opt) < 1e-12:
                                # tie-breaking: smaller i, then smaller j
                                if i < (i_orig if 'i_orig' in locals() else i) or (i == (i_orig if 'i_orig' in locals() else i) and j < (j_orig if 'j_orig' in locals() else j)):
                                    # not sure we need to store best i,j; just keep first found
                                    pass
                    if found:
                        routes[longest_idx] = best_route
                        new_max = compute_max()
                        if new_max < best_max - 1e-12:
                            best_max = new_max
                            best_routes = copy_routes()
                            report_best_vrp(best_routes)
                        improved = True
            elif op == 'relocate':
                # Best relocate from longest route to any other
                src_route = routes[longest_idx]
                if len(src_route) <= 2:
                    continue
                best_improvement = 0.0
                best_move = None
                current_max = compute_max()
                for pos_i in range(1, len(src_route) - 1):
                    cust = src_route[pos_i]
                    new_src = src_route[:pos_i] + src_route[pos_i+1:]
                    dist_src = route_dist(new_src)
                    for dst_idx in range(len(routes)):
                        if dst_idx == longest_idx:
                            continue
                        dst_route = routes[dst_idx]
                        for pos_j in range(1, len(dst_route)):
                            new_dst = dst_route[:pos_j] + [cust] + dst_route[pos_j:]
                            dist_dst = route_dist(new_dst)
                            other_dists = [route_dist(r) for i, r in enumerate(routes) if i not in (longest_idx, dst_idx)]
                            new_max = max([dist_src, dist_dst] + other_dists)
                            improvement = current_max - new_max
                            if improvement > best_improvement + 1e-12:
                                best_improvement = improvement
                                best_move = (cust, pos_i, dst_idx, pos_j, new_src, new_dst)
                            elif abs(improvement - best_improvement) < 1e-12 and best_move is not None:
                                (ocust, opi, odst_idx, opj, _, _) = best_move
                                if cust < ocust or (cust == ocust and pos_i < opi) or (cust == ocust and pos_i == opi and dst_idx < odst_idx) or (cust == ocust and pos_i == opi and dst_idx == odst_idx and pos_j < opj):
                                    best_improvement = improvement
                                    best_move = (cust, pos_i, dst_idx, pos_j, new_src, new_dst)
                if best_move is not None and best_improvement > 0:
                    cust, pos_i, dst_idx, pos_j, new_src, new_dst = best_move
                    routes[longest_idx] = new_src
                    routes[dst_idx] = new_dst
                    new_max = compute_max()
                    if new_max < best_max - 1e-12:
                        best_max = new_max
                        best_routes = copy_routes()
                        report_best_vrp(best_routes)
                    improved = True
            elif op == 'swap':
                # Best swap between longest and any other route
                src_route = routes[longest_idx]
                if len(src_route) <= 2:
                    continue
                best_improvement = 0.0
                best_move = None
                current_max = compute_max()
                for pos_i in range(1, len(src_route) - 1):
                    cust_i = src_route[pos_i]
                    for dst_idx in range(len(routes)):
                        if dst_idx == longest_idx:
                            continue
                        dst_route = routes[dst_idx]
                        if len(dst_route) <= 2:
                            continue
                        for pos_j in range(1, len(dst_route) - 1):
                            cust_j = dst_route[pos_j]
                            new_src = src_route[:pos_i] + [cust_j] + src_route[pos_i+1:]
                            new_dst = dst_route[:pos_j] + [cust_i] + dst_route[pos_j+1:]
                            dist_src = route_dist(new_src)
                            dist_dst = route_dist(new_dst)
                            other_dists = [route_dist(r) for i, r in enumerate(routes) if i not in (longest_idx, dst_idx)]
                            new_max = max([dist_src, dist_dst] + other_dists)
                            improvement = current_max - new_max
                            if improvement > best_improvement + 1e-12:
                                best_improvement = improvement
                                best_move = (pos_i, dst_idx, pos_j, new_src, new_dst)
                            elif abs(improvement - best_improvement) < 1e-12 and best_move is not None:
                                (opi, odst_idx, opj, _, _) = best_move
                                if cust_i < src_route[opi] or (cust_i == src_route[opi] and pos_i < opi) or (cust_i == src_route[opi] and pos_i == opi and dst_idx < odst_idx) or (cust_i == src_route[opi] and pos_i == opi and dst_idx == odst_idx and pos_j < opj):
                                    best_improvement = improvement
                                    best_move = (pos_i, dst_idx, pos_j, new_src, new_dst)
                if best_move is not None and best_improvement > 0:
                    pos_i, dst_idx, pos_j, new_src, new_dst = best_move
                    routes[longest_idx] = new_src
                    routes[dst_idx] = new_dst
                    new_max = compute_max()
                    if new_max < best_max - 1e-12:
                        best_max = new_max
                        best_routes = copy_routes()
                        report_best_vrp(best_routes)
                    improved = True
        if not improved:
            break
    return best_routes