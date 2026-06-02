import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    customers = list(range(1, n))
    
    # find farthest customer from depot to start TSP
    depot = 0
    farthest = max(customers, key=lambda c: distance_matrix[depot][c])
    
    # build giant TSP tour via nearest neighbor
    giant = []
    current = farthest
    giant.append(current)
    remaining = set(customers)
    remaining.remove(current)
    while remaining:
        next_cust = min(remaining, key=lambda c: distance_matrix[current][c])
        giant.append(next_cust)
        remaining.remove(next_cust)
        current = next_cust
    L = len(giant)
    
    # DP split minimize max route distance
    dist0 = [distance_matrix[0][c] for c in giant]
    dist_to0 = [distance_matrix[c][0] for c in giant]
    edge = [distance_matrix[giant[i]][giant[i+1]] for i in range(L-1)]
    pref = [0]*L
    for i in range(1, L):
        pref[i] = pref[i-1] + edge[i-1]
    def seg_cost(i, j):
        return dist0[i] + (pref[j] - pref[i]) + dist_to0[j]
    INF = 1e18
    dp = [[INF]*(L+1) for _ in range(truck_count+1)]
    prev = [[-1]*(L+1) for _ in range(truck_count+1)]
    dp[0][0] = 0
    for k in range(1, truck_count+1):
        for i in range(k, L+1):
            best_val = INF
            best_j = -1
            for j in range(k-1, i):
                if dp[k-1][j] < INF:
                    cand = max(dp[k-1][j], seg_cost(j, i-1))
                    if cand < best_val:
                        best_val = cand
                        best_j = j
            dp[k][i] = best_val
            prev[k][i] = best_j
    split = []
    k = truck_count
    i = L
    while k > 0 and i > 0:
        j = prev[k][i]
        split.append(j)
        i = j
        k -= 1
    split.reverse()
    if len(split) < truck_count:
        split = [0] + [L*(r+1)//truck_count for r in range(truck_count)]
    split = split[:truck_count] + [L]
    routes = []
    for r in range(truck_count):
        start = split[r]
        end = split[r+1]
        route_custs = giant[start:end]
        routes.append([0] + route_custs + [0])
    while len(routes) < truck_count:
        routes.append([0,0])
    
    def route_dist(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    try:
        report_best_vrp(best_routes)
    except:
        pass
    
    # local search auxiliary
    def two_opt_route(route):
        if len(route) <= 3:
            return route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    old = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                    new = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                    if new < old - 1e-10:
                        route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        improved = True
        return route
    
    def relocate_from_longest(routes):
        # relocate one customer from longest route to another route if improves max
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(len(routes)), key=lambda i: dists[i])
        if dists[longest_idx] <= 0:
            return routes
        route_long = routes[longest_idx]
        for cust in route_long[1:-1]:
            for to_idx in range(len(routes)):
                if to_idx == longest_idx:
                    continue
                route_to = routes[to_idx]
                for pos in range(1, len(route_to)):
                    new_route_long = [0] + [c for c in route_long[1:-1] if c != cust] + [0]
                    if len(new_route_long) == 2:
                        new_route_long = [0,0]
                    new_route_to = route_to[:pos] + [cust] + route_to[pos:]
                    new_dist_long = route_dist(new_route_long)
                    new_dist_to = route_dist(new_route_to)
                    other_dists = [route_dist(routes[k]) for k in range(len(routes)) if k not in (longest_idx, to_idx)]
                    new_max = max([new_dist_long, new_dist_to] + other_dists)
                    old_max = max(dists)
                    if new_max < old_max - 1e-10:
                        routes[longest_idx] = new_route_long
                        routes[to_idx] = new_route_to
                        return routes, new_max
        return routes, max(dists)
    
    # main improvement loop
    for cycle in range(10):
        # intra-route 2-opt on each route
        for idx in range(truck_count):
            if len(routes[idx]) > 3:
                routes[idx] = two_opt_route(routes[idx])
        # inter-route relocate from longest
        routes, cur_max = relocate_from_longest(routes)
        if cur_max < best_max - 1e-10:
            best_max = cur_max
            best_routes = [list(r) for r in routes]
            try:
                report_best_vrp(best_routes)
            except:
                pass
        
        # ruin and recreate
        all_custs = [c for c in customers]
        random.shuffle(all_custs)
        num_remove = max(1, int(0.2 * (n-1)))
        # identify longest route
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(truck_count), key=lambda i: dists[i])
        longest_custs = [c for c in routes[longest_idx][1:-1]]
        removed = []
        num_from_longest = min(len(longest_custs), int(num_remove * 0.5))
        random.shuffle(longest_custs)
        removed.extend(longest_custs[:num_from_longest])
        remaining_custs = [c for c in all_custs if c not in removed]
        random.shuffle(remaining_custs)
        needed = num_remove - len(removed)
        if needed > 0:
            removed.extend(remaining_custs[:needed])
        # remove from routes
        for r in routes:
            r[:] = [0] + [c for c in r[1:-1] if c not in removed] + [0]
        # greedy reinsertion minimizing max route distance
        random.shuffle(removed)
        for cust in removed:
            best_route = -1
            best_pos = -1
            best_new_max = float('inf')
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_dist(new_route)
                    other_max = max(route_dist(routes[k]) for k in range(truck_count) if k != r_idx)
                    new_max = max(new_dist, other_max)
                    if new_max < best_new_max - 1e-10:
                        best_new_max = new_max
                        best_route = r_idx
                        best_pos = pos
            if best_route != -1:
                routes[best_route] = routes[best_route][:best_pos] + [cust] + routes[best_route][best_pos:]
    
    # finalize
    used = set()
    for r in best_routes:
        for c in r:
            if c != 0:
                used.add(c)
    missing = [c for c in customers if c not in used]
    if missing:
        best_routes[0] = [0] + best_routes[0][1:-1] + missing + [0]
    while len(best_routes) < truck_count:
        best_routes.append([0,0])
    return best_routes