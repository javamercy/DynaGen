import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # --- Build giant tour via nearest neighbor ---
    current = 0
    tour = [0]
    unvisited = set(customers)
    while unvisited:
        next_node = min(unvisited, key=lambda x: (distance_matrix[current][x], x))
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    tour.append(0)  # return to depot
    # Customer sequence (order of first visit, excluding final depot)
    cust_seq = tour[1:-1]
    m = len(cust_seq)

    # Precompute distances for segments
    dep_to_cust = [distance_matrix[0][c] for c in cust_seq]
    cust_to_dep = [distance_matrix[c][0] for c in cust_seq]
    cum = [0.0] * m
    for i in range(1, m):
        cum[i] = cum[i-1] + distance_matrix[cust_seq[i-1]][cust_seq[i]]

    # --- DP to split into truck_count routes minimizing max route distance ---
    def segment_distance(i, j):
        # i inclusive, j exclusive: customers i..j-1
        if i >= j:
            return 0.0
        # route: depot -> [i..j-1] -> depot
        return dep_to_cust[i] + (cum[j-1] - cum[i]) + cust_to_dep[j-1]

    INF = 1e100
    # dp[k][i] = min max distance for first i customers using k routes
    # We'll use list of arrays for clarity
    dp = [np.full(m+1, INF) for _ in range(truck_count+1)]
    prev = [np.full(m+1, -1, dtype=int) for _ in range(truck_count+1)]
    dp[0][0] = 0.0
    for k in range(1, truck_count+1):
        for i in range(k, m+1):  # need at least i customers for k routes
            best_val = INF
            best_j = -1
            for j in range(k-1, i):
                seg = segment_distance(j, i)
                cand = max(dp[k-1][j], seg)
                if cand < best_val - 1e-12 or (abs(cand - best_val) < 1e-12 and j < best_j):
                    best_val = cand
                    best_j = j
            dp[k][i] = best_val
            prev[k][i] = best_j

    # Reconstruct routes
    routes = []
    i = m
    k = truck_count
    while k > 0:
        j = prev[k][i]
        # customers from j to i-1
        seg_cust = cust_seq[j:i]
        route = [0] + seg_cust + [0]
        routes.append(route)
        i = j
        k -= 1
    # Since we backtracked, routes are in reverse order; reverse to get original order
    routes.reverse()
    # If truck_count > m, add empty routes (but DP ensures each route has at least one customer)
    while len(routes) < truck_count:
        routes.append([0, 0])

    # Report initial solution
    current_max = max(route_dist(r) for r in routes)
    report_best_vrp(routes)

    # --- Threshold accepting improvement ---
    max_iter = n * truck_count
    threshold = 0.3 * current_max
    cooling = 0.95
    for iteration in range(max_iter):
        improved = False
        # Find longest route
        max_idx = max(range(truck_count), key=lambda i: route_dist(routes[i]))
        long_route = routes[max_idx]
        # Relocate customers from longest route
        for pos, cust in enumerate(long_route[1:-1]):
            new_long = long_route[:pos+1] + long_route[pos+2:]
            for t_idx in range(truck_count):
                if t_idx == max_idx:
                    continue
                target_route = routes[t_idx]
                for ins_pos in range(1, len(target_route)):
                    new_target = target_route[:ins_pos] + [cust] + target_route[ins_pos:]
                    new_max_val = max(route_dist(new_long), route_dist(new_target))
                    for r_idx in range(truck_count):
                        if r_idx not in (max_idx, t_idx):
                            new_max_val = max(new_max_val, route_dist(routes[r_idx]))
                    if new_max_val <= current_max + threshold:
                        if new_max_val < current_max:
                            current_max = new_max_val
                            routes[max_idx] = new_long
                            routes[t_idx] = new_target
                            report_best_vrp(routes)
                            improved = True
                            break
                        else:
                            routes[max_idx] = new_long
                            routes[t_idx] = new_target
                            current_max = new_max_val
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            threshold *= cooling
            continue
        # Exchange customers between longest route and others
        for pos, cust in enumerate(long_route[1:-1]):
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for opos, ocust in enumerate(other_route[1:-1]):
                    new_long = long_route[:pos+1] + [ocust] + long_route[pos+2:]
                    new_other = other_route[:opos+1] + [cust] + other_route[opos+2:]
                    new_max_val = max(route_dist(new_long), route_dist(new_other))
                    for r_idx in range(truck_count):
                        if r_idx not in (max_idx, other_idx):
                            new_max_val = max(new_max_val, route_dist(routes[r_idx]))
                    if new_max_val <= current_max + threshold:
                        if new_max_val < current_max:
                            current_max = new_max_val
                            routes[max_idx] = new_long
                            routes[other_idx] = new_other
                            report_best_vrp(routes)
                            improved = True
                            break
                        else:
                            routes[max_idx] = new_long
                            routes[other_idx] = new_other
                            current_max = new_max_val
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
        threshold *= cooling

    # Intra-route 2-opt
    for r_idx in range(truck_count):
        route = routes[r_idx]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    a = route[i-1]
                    b = route[i]
                    c = route[j]
                    d = route[j+1]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new_dist = distance_matrix[a][c] + distance_matrix[b][d]
                    if new_dist < old - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break
    return routes

def route_dist(r):
    d = 0
    for i in range(len(r)-1):
        d += distance_matrix[r[i]][r[i+1]]
    return d