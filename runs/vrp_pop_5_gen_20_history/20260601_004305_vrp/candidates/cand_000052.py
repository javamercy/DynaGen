import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # --- TSP tour using nearest neighbor (deterministic) ---
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

    # --- DP minimax split (minimize max route distance) ---
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
            total += distance_matrix[route[a], route[a + 1]]
        return total

    def compute_max():
        return max(route_dist(r) for r in routes)

    best_routes = [list(r) for r in routes]
    best_max = compute_max()
    report_best_vrp(best_routes)

    # --- Improvement: iterative relocation from max route ---
    max_iter = m  # finite bound
    for _ in range(max_iter):
        # Identify route with maximum distance (deterministic tie: smallest index)
        dists = [route_dist(r) for r in routes]
        max_val = max(dists)
        max_idx = dists.index(max_val)  # first occurrence
        max_route = routes[max_idx]

        revised = False
        best_new_max = max_val
        best_move = None  # (cust, other_idx, pos, new_max, new_route_max, new_route_other)

        # Evaluate each customer in the max route
        for cust in max_route[1:-1]:  # skip depot
            # Temporarily remove cust from max route
            temp_route_max = [x for x in max_route if x != cust]
            if len(temp_route_max) < 2:
                continue
            new_dist_max = route_dist(temp_route_max)

            # For each other route, find best insertion position for cust using regret-2
            # Compute best and second best resulting max after insertion
            best_ins = (math.inf, -1, -1)
            second_ins = (math.inf, -1, -1)
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                # trial insertions
                for pos in range(1, len(other_route)):
                    new_route_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_dist_other = route_dist(new_route_other)
                    # compute new max ignoring the max route (which lost cust)
                    # we need all routes except max_idx, so take dists of unchanged routes, replace max with new_dist_max, and replace other with new_dist_other
                    candidate_dists = [new_dist_max if i == max_idx else (new_dist_other if i == other_idx else dists[i]) for i in range(truck_count)]
                    cand_max = max(candidate_dists)
                    if cand_max < best_ins[0]:
                        second_ins = best_ins
                        best_ins = (cand_max, other_idx, pos)
                    elif cand_max < second_ins[0]:
                        second_ins = (cand_max, other_idx, pos)

            if best_ins[0] == math.inf:
                continue
            regret = second_ins[0] - best_ins[0]
            # Consider move if it improves overall max, with tie-breaking by regret (larger regret better) then customer index, then route index, then position
            if best_ins[0] < best_new_max:
                best_new_max = best_ins[0]
                best_move = (cust, best_ins[1], best_ins[2], temp_route_max, routes[best_ins[1]][:best_ins[2]] + [cust] + routes[best_ins[1]][best_ins[2]:])
            elif best_ins[0] == best_new_max:
                # tie-break: larger regret, smaller customer index, smaller route index, smaller position
                if best_move is None:
                    best_move = (cust, best_ins[1], best_ins[2], temp_route_max, routes[best_ins[1]][:best_ins[2]] + [cust] + routes[best_ins[1]][best_ins[2]:])
                else:
                    prev_regret = best_ins[0] - best_new_max  # actually second - first, but since first == best_new_max, regret = second - best_new_max
                    # compute regret for current best_move? Not needed, we use regret of candidate
                    regret_cand = second_ins[0] - best_ins[0]
                    # compare by regret (higher better)
                    if regret_cand > regret:
                        best_move = (cust, best_ins[1], best_ins[2], temp_route_max, routes[best_ins[1]][:best_ins[2]] + [cust] + routes[best_ins[1]][best_ins[2]:])
                    elif regret_cand == regret:
                        # smaller customer index
                        if cust < best_move[0]:
                            best_move = (cust, best_ins[1], best_ins[2], temp_route_max, routes[best_ins[1]][:best_ins[2]] + [cust] + routes[best_ins[1]][best_ins[2]:])
                        elif cust == best_move[0]:
                            # smaller other route index
                            if best_ins[1] < best_move[1]:
                                best_move = (cust, best_ins[1], best_ins[2], temp_route_max, routes[best_ins[1]][:best_ins[2]] + [cust] + routes[best_ins[1]][best_ins[2]:])
                            elif best_ins[1] == best_move[1] and best_ins[2] < best_move[2]:
                                best_move = (cust, best_ins[1], best_ins[2], temp_route_max, routes[best_ins[1]][:best_ins[2]] + [cust] + routes[best_ins[1]][best_ins[2]:])

        if best_move is not None and best_new_max < best_max:
            cust, other_idx, pos, new_max_route, new_other_route = best_move
            routes[max_idx] = new_max_route
            routes[other_idx] = new_other_route
            best_max = best_new_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
            revised = True

        if not revised:
            break

    return best_routes