import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        for _ in range(truck_count - m):
            routes.append([0, 0])
        return routes

    # TSP tour using nearest neighbor
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

    custs = tour
    k = truck_count
    # Precompute segment distances
    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0, custs[l]]
        for r in range(l + 1, m + 1):
            if r > l + 1:
                acc += distance_matrix[custs[r-2], custs[r-1]]
            if r == l + 1:
                route_dist = distance_matrix[0, custs[l]] + distance_matrix[custs[l], 0]
            else:
                route_dist = acc + distance_matrix[custs[r-1], 0]
            seg_dist[l][r] = route_dist

    # DP minimax split
    INF = math.inf
    dp = [[INF] * (k + 1) for _ in range(m + 1)]
    choice = [[-1] * (k + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    for i in range(1, m + 1):
        for t in range(1, min(i, k) + 1):
            best_val = INF
            best_j = -1
            for j in range(t - 1, i):
                if dp[j][t-1] < INF:
                    cand = max(dp[j][t-1], seg_dist[j][i])
                    if cand < best_val:
                        best_val = cand
                        best_j = j
            dp[i][t] = best_val
            choice[i][t] = best_j

    # Reconstruct routes
    routes = []
    i = m
    t = k
    while t > 0:
        j = choice[i][t]
        seg = custs[j:i]
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

    def compute_max_and_total():
        maxd = 0
        total = 0
        for r in routes:
            d = route_dist(r)
            total += d
            if d > maxd:
                maxd = d
        return maxd, total

    best_max, best_total = compute_max_and_total()
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)

    max_iter = n * 10
    for _ in range(max_iter):
        current_max, current_total = compute_max_and_total()
        best_improvement = 0.0
        best_total_improvement = 0.0
        best_move = None

        # Evaluate all 2-opt moves
        for ri in range(truck_count):
            route = routes[ri]
            if len(route) <= 3:
                continue
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    old_dist = route_dist(route)
                    delta = old_dist - new_dist
                    new_max = max(current_max, new_dist) if new_dist > current_max else current_max
                    # Actually need to check if the max changes: could be that old_dist was max, now new_dist < old_dist
                    # But simplest: compute new max for the entire set
                    # Let's just compute new max and total
                    temp_routes = [list(r) for r in routes]
                    temp_routes[ri] = new_route
                    temp_max = max(route_dist(r) for r in temp_routes)
                    temp_total = sum(route_dist(r) for r in temp_routes)
                    improv = current_max - temp_max
                    total_improv = current_total - temp_total
                    if improv > best_improvement or (improv == best_improvement and total_improv > best_total_improvement):
                        best_improvement = improv
                        best_total_improvement = total_improv
                        best_move = ('2opt', ri, i, j, new_route)

        # Evaluate all relocate moves
        for ri in range(truck_count):
            route = routes[ri]
            for pos in range(1, len(route) - 1):
                cust = route[pos]
                for rj in range(truck_count):
                    if rj == ri:
                        continue
                    dest_route = routes[rj]
                    for dp in range(1, len(dest_route)):
                        new_src = route[:pos] + route[pos+1:]
                        new_dst = dest_route[:dp] + [cust] + dest_route[dp:]
                        temp_routes = [list(r) for r in routes]
                        temp_routes[ri] = new_src
                        temp_routes[rj] = new_dst
                        temp_max = max(route_dist(r) for r in temp_routes)
                        temp_total = sum(route_dist(r) for r in temp_routes)
                        improv = current_max - temp_max
                        total_improv = current_total - temp_total
                        if improv > best_improvement or (improv == best_improvement and total_improv > best_total_improvement):
                            best_improvement = improv
                            best_total_improvement = total_improv
                            best_move = ('relocate', ri, pos, rj, dp, cust)

        # Evaluate all swap moves
        for ri in range(truck_count):
            route_i = routes[ri]
            for pi in range(1, len(route_i) - 1):
                cust_i = route_i[pi]
                for rj in range(ri + 1, truck_count):
                    route_j = routes[rj]
                    for pj in range(1, len(route_j) - 1):
                        cust_j = route_j[pj]
                        new_i = route_i[:pi] + [cust_j] + route_i[pi+1:]
                        new_j = route_j[:pj] + [cust_i] + route_j[pj+1:]
                        temp_routes = [list(r) for r in routes]
                        temp_routes[ri] = new_i
                        temp_routes[rj] = new_j
                        temp_max = max(route_dist(r) for r in temp_routes)
                        temp_total = sum(route_dist(r) for r in temp_routes)
                        improv = current_max - temp_max
                        total_improv = current_total - temp_total
                        if improv > best_improvement or (improv == best_improvement and total_improv > best_total_improvement):
                            best_improvement = improv
                            best_total_improvement = total_improv
                            best_move = ('swap', ri, pi, rj, pj)

        if best_move is None or best_improvement <= 0:
            break
        # Apply best move
        move_type = best_move[0]
        if move_type == '2opt':
            _, ri, i, j, new_route = best_move
            routes[ri] = new_route
        elif move_type == 'relocate':
            _, ri, pos, rj, dp, cust = best_move
            route = routes[ri]
            routes[ri] = route[:pos] + route[pos+1:]
            routes[rj] = routes[rj][:dp] + [cust] + routes[rj][dp:]
        elif move_type == 'swap':
            _, ri, pi, rj, pj = best_move
            routes[ri][pi], routes[rj][pj] = routes[rj][pj], routes[ri][pi]
        new_max, new_total = compute_max_and_total()
        if new_max < best_max or (new_max == best_max and new_total < best_total):
            best_max = new_max
            best_total = new_total
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    return best_routes