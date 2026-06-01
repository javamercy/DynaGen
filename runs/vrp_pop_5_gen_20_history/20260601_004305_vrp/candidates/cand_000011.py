import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = len(distance_matrix)
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)] + [[0, 0]] * (truck_count - (n - 1))
        report_best_vrp(routes)
        return routes
    
    # Step 1: Build giant TSP tour via cheapest insertion
    customers = list(range(1, n))
    tour = [0, 0]  # start and end at depot
    inserted = set()
    while len(inserted) < len(customers):
        best_cust = None
        best_pos = None
        best_cost = float('inf')
        for c in customers:
            if c in inserted:
                continue
            for pos in range(1, len(tour)):
                # cost increase of inserting c between tour[pos-1] and tour[pos]
                delta = (distance_matrix[tour[pos-1]][c] +
                         distance_matrix[c][tour[pos]] -
                         distance_matrix[tour[pos-1]][tour[pos]])
                if delta < best_cost or (delta == best_cost and c < best_cust):
                    best_cost = delta
                    best_cust = c
                    best_pos = pos
        # insert best customer at best_pos
        tour.insert(best_pos, best_cust)
        inserted.add(best_cust)
    # tour now is [0, c1, c2, ..., cm, 0]
    perm = tour[1:-1]  # customer sequence
    m = len(perm)
    
    # Precompute cost of segment from i to j (inclusive) in perm
    # cost[i][j] = distance(0, perm[i]) + sum_{k=i}^{j-1} dist(perm[k], perm[k+1]) + distance(perm[j], 0)
    cost = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i, m):
            seg_cost = distance_matrix[0][perm[i]]
            for k in range(i, j):
                seg_cost += distance_matrix[perm[k]][perm[k+1]]
            seg_cost += distance_matrix[perm[j]][0]
            cost[i][j] = seg_cost
    
    # DP: dp[r][i] = min max for first i+1 customers (0..i) using r routes
    max_val = 1e15
    dp = np.full((truck_count+1, m), max_val)
    # base: r=1
    for i in range(m):
        dp[1][i] = cost[0][i]
    # fill
    for r in range(2, truck_count+1):
        for i in range(r-1, m):
            for j in range(r-2, i):  # j is end of previous segment, earlier segment covers 0..j
                cand = max(dp[r-1][j], cost[j+1][i])
                if cand < dp[r][i]:
                    dp[r][i] = cand
    
    # Reconstruction
    cuts = []
    cur_r = truck_count
    cur_i = m - 1
    while cur_r > 1:
        for j in range(cur_r-2, cur_i):
            if abs(dp[cur_r][cur_i] - max(dp[cur_r-1][j], cost[j+1][cur_i])) < 1e-9:
                cuts.append(j+1)  # start index of last segment
                cur_i = j
                cur_r -= 1
                break
    cuts.append(0)  # first segment starts at 0
    cuts.reverse()
    # cuts now are start indices of each segment
    routes = []
    for s_idx in range(truck_count):
        start = cuts[s_idx]
        if s_idx == truck_count - 1:
            end = m
        else:
            end = cuts[s_idx+1]
        if start == end:
            # empty route
            routes.append([0, 0])
        else:
            route = [0] + perm[start:end] + [0]
            routes.append(route)
    
    # Report initial solution
    def route_dist(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    costs = [route_dist(r) for r in routes]
    best_max = max(costs)
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)
    
    # Improvement: 2-opt intra-route and relocate inter-route
    def improve():
        nonlocal best_routes, best_max, routes
        improved = True
        for _ in range(10):  # bounded passes
            if not improved:
                break
            improved = False
            # 2-opt each route
            for idx in range(len(routes)):
                route = routes[idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        if j - i == 1:
                            continue
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_dist(new_route)
                        if new_dist < route_dist(route):
                            routes[idx] = new_route
                            improved = True
            # Relocate customers
            for idx in range(len(routes)):
                route = routes[idx]
                for pos in range(1, len(route)-1):
                    cust = route[pos]
                    # remove cust
                    temp_route = route[:pos] + route[pos+1:]
                    new_route_cost = route_dist(temp_route)
                    best_other_idx = None
                    best_other_pos = None
                    best_new_max = float('inf')
                    for other_idx in range(len(routes)):
                        if other_idx == idx:
                            continue
                        other = routes[other_idx]
                        for p in range(1, len(other)):
                            new_other = other[:p] + [cust] + other[p:]
                            new_other_dist = route_dist(new_other)
                            # compute candidate max
                            cand_max = max(new_route_cost, new_other_dist)
                            for k in range(len(routes)):
                                if k != idx and k != other_idx:
                                    cand_max = max(cand_max, route_dist(routes[k]))
                            if cand_max < best_new_max:
                                best_new_max = cand_max
                                best_other_idx = other_idx
                                best_other_pos = p
                    if best_other_idx is not None and best_new_max < max(route_dist(routes[idx]), route_dist(routes[best_other_idx])):
                        # commit move
                        routes[idx] = temp_route
                        routes[best_other_idx] = routes[best_other_idx][:best_other_pos] + [cust] + routes[best_other_idx][best_other_pos:]
                        improved = True
            # Update best
            costs = [route_dist(r) for r in routes]
            current_max = max(costs)
            if current_max < best_max:
                best_max = current_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
    
    improve()
    return best_routes