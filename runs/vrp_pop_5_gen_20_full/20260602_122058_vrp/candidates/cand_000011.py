import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # Helper: compute route distance
    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d

    # --- Initial Construction: farthest-first clustering + cheapest insertion ---
    # Seed selection
    seeds = []
    first_seed = max(range(1, n), key=lambda i: (distance_matrix[0][i], -i))
    seeds.append(first_seed)
    for _ in range(1, truck_count):
        best_min_dist = -1
        best_node = None
        for node in range(1, n):
            if node in seeds:
                continue
            min_dist = min(distance_matrix[node][s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and node < best_node):
                best_min_dist = min_dist
                best_node = node
        if best_node is None:
            break
        seeds.append(best_node)

    # Assign customers to nearest seed
    clusters = {s: [] for s in seeds}
    for node in range(1, n):
        if node in seeds:
            clusters[node].append(node)
        else:
            nearest = min(seeds, key=lambda s: (distance_matrix[node][s], s))
            clusters[nearest].append(node)

    # Build routes via cheapest insertion per cluster
    routes = []
    for seed in seeds:
        cluster = list(clusters[seed])
        # start with depot-seed-depot
        route = [0, seed, 0]
        cluster.remove(seed)
        while cluster:
            best_cost = float('inf')
            best_cust = None
            best_pos = None
            for cust in cluster:
                for pos in range(1, len(route)):
                    delta = distance_matrix[route[pos-1]][cust] + distance_matrix[cust][route[pos]] - distance_matrix[route[pos-1]][route[pos]]
                    if delta < best_cost - 1e-12 or (abs(delta - best_cost) < 1e-12 and cust < best_cust):
                        best_cost = delta
                        best_cust = cust
                        best_pos = pos
            route = route[:best_pos] + [best_cust] + route[best_pos:]
            cluster.remove(best_cust)
        routes.append(route)

    # Fill empty trucks if needed
    while len(routes) < truck_count:
        routes.append([0, 0])

    current_max = max(route_dist(r) for r in routes)
    report_best_vrp(routes)

    # --- Balancing phase: move customers from longest to shortest ---
    for _ in range(n):
        # Find longest and shortest routes by distance
        dists = [route_dist(r) for r in routes]
        longest_idx = max(range(truck_count), key=lambda i: (dists[i], i))
        shortest_idx = min(range(truck_count), key=lambda i: (dists[i], -i))
        if longest_idx == shortest_idx:
            break
        long_route = routes[longest_idx]
        short_route = routes[shortest_idx]
        # Try to move a customer from longest to shortest
        best_delta = 0
        best_move = None
        for pos, cust in enumerate(long_route[1:-1]):
            new_long = long_route[:pos+1] + long_route[pos+2:]
            # Find best insertion point in short_route
            for ins in range(1, len(short_route)):
                new_short = short_route[:ins] + [cust] + short_route[ins:]
                new_max = max(route_dist(new_long), route_dist(new_short), max(dists[i] for i in range(truck_count) if i not in (longest_idx, shortest_idx)))
                delta = current_max - new_max
                if delta > best_delta + 1e-12 or (abs(delta - best_delta) < 1e-12 and (cust < best_move[0] or (cust == best_move[0] and ins < best_move[2]))):
                    best_delta = delta
                    best_move = (cust, pos, ins)
        if best_move and best_delta > 0:
            cust, pos, ins = best_move
            long_route = routes[longest_idx][:pos+1] + routes[longest_idx][pos+2:]
            new_short = routes[shortest_idx][:ins] + [cust] + routes[shortest_idx][ins:]
            routes[longest_idx] = long_route
            routes[shortest_idx] = new_short
            current_max = max(route_dist(r) for r in routes)
            report_best_vrp(routes)
        else:
            break

    # --- Variable Neighborhood Descent ---
    max_iter = n * truck_count
    for outer in range(max_iter):
        improved_outer = False
        # Neighborhood cycle: 2opt, relocate, exchange, cross
        for nh in range(4):
            improved = False
            if nh == 0:  # intra-route 2-opt
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            a = route[i-1]
                            b = route[i]
                            c = route[j]
                            d = route[j+1]
                            old = distance_matrix[a][b] + distance_matrix[c][d]
                            newd = distance_matrix[a][c] + distance_matrix[b][d]
                            if newd < old - 1e-12:
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                routes[r_idx] = new_route
                                current_max = max(route_dist(r) for r in routes)
                                report_best_vrp(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            elif nh == 1:  # relocate: move one customer from one route to another
                best_rel = None
                best_new_max = float('inf')
                for i in range(truck_count):
                    for pos, cust in enumerate(routes[i][1:-1]):
                        new_i = routes[i][:pos+1] + routes[i][pos+2:]
                        for j in range(truck_count):
                            if j == i and len(new_i) <= 2:
                                continue
                            target = routes[j]
                            for ins in range(1, len(target)):
                                new_j = target[:ins] + [cust] + target[ins:]
                                new_dist = max(route_dist(new_i), route_dist(new_j))
                                for k in range(truck_count):
                                    if k not in (i, j):
                                        new_dist = max(new_dist, route_dist(routes[k]))
                                if new_dist < best_new_max - 1e-12 or (abs(new_dist - best_new_max) < 1e-12 and (cust < best_rel[0] or (cust == best_rel[0] and i < best_rel[1]))):
                                    best_new_max = new_dist
                                    best_rel = (cust, pos, i, j, ins)
                if best_rel and best_new_max < current_max - 1e-12:
                    cust, pos, i, j, ins = best_rel
                    routes[i] = routes[i][:pos+1] + routes[i][pos+2:]
                    routes[j] = routes[j][:ins] + [cust] + routes[j][ins:]
                    current_max = best_new_max
                    report_best_vrp(routes)
                    improved = True
            elif nh == 2:  # exchange: swap two customers
                best_ex = None
                best_new_max = float('inf')
                for i in range(truck_count):
                    for pos_i, cust_i in enumerate(routes[i][1:-1]):
                        for j in range(i, truck_count):
                            start = 0 if i == j else 1
                            for pos_j in range(start, len(routes[j])-1):
                                if i == j and pos_i >= pos_j:
                                    continue
                                cust_j = routes[j][pos_j]
                                if i == j:
                                    # same route: swap two customers
                                    if pos_i == pos_j:
                                        continue
                                    new_route = list(routes[i])
                                    new_route[pos_i], new_route[pos_j] = cust_j, cust_i
                                    new_dist = max(route_dist(new_route))
                                    for k in range(truck_count):
                                        if k != i:
                                            new_dist = max(new_dist, route_dist(routes[k]))
                                    if new_dist < best_new_max - 1e-12 or (abs(new_dist - best_new_max) < 1e-12 and (min(cust_i, cust_j) < min(best_ex[0], best_ex[1]) if best_ex else True)):
                                        best_new_max = new_dist
                                        best_ex = (i, pos_i, cust_i, j, pos_j, cust_j, new_route)
                                else:
                                    # different routes
                                    new_i = routes[i][:pos_i+1] + [cust_j] + routes[i][pos_i+2:]
                                    new_j = routes[j][:pos_j+1] + [cust_i] + routes[j][pos_j+2:]
                                    new_dist = max(route_dist(new_i), route_dist(new_j))
                                    for k in range(truck_count):
                                        if k not in (i, j):
                                            new_dist = max(new_dist, route_dist(routes[k]))
                                    if new_dist < best_new_max - 1e-12 or (abs(new_dist - best_new_max) < 1e-12 and (min(cust_i, cust_j) < min(best_ex[0], best_ex[1]) if best_ex else True)):
                                        best_new_max = new_dist
                                        best_ex = (i, pos_i, cust_i, j, pos_j, cust_j, new_i, new_j)
                if best_ex and best_new_max < current_max - 1e-12:
                    if len(best_ex) == 7:  # same route
                        i, pos_i, cust_i, j, pos_j, cust_j, new_route = best_ex
                        routes[i] = new_route
                    else:
                        i, pos_i, cust_i, j, pos_j, cust_j, new_i, new_j = best_ex
                        routes[i] = new_i
                        routes[j] = new_j
                    current_max = best_new_max
                    report_best_vrp(routes)
                    improved = True
            elif nh == 3:  # cross: swap segments of length up to 2 between two routes
                best_cross = None
                best_new_max = float('inf')
                for i in range(truck_count):
                    for j in range(i+1, truck_count):
                        route_i = routes[i]
                        route_j = routes[j]
                        # Consider segment lengths 1 and 2
                        for len_i in [1, 2]:
                            for len_j in [1, 2]:
                                for start_i in range(1, len(route_i)-len_i):
                                    for start_j in range(1, len(route_j)-len_j):
                                        seg_i = route_i[start_i:start_i+len_i]
                                        seg_j = route_j[start_j:start_j+len_j]
                                        new_i = route_i[:start_i] + seg_j + route_i[start_i+len_i:]
                                        new_j = route_j[:start_j] + seg_i + route_j[start_j+len_j:]
                                        new_dist = max(route_dist(new_i), route_dist(new_j))
                                        for k in range(truck_count):
                                            if k not in (i, j):
                                                new_dist = max(new_dist, route_dist(routes[k]))
                                        if new_dist < best_new_max - 1e-12 or (abs(new_dist - best_new_max) < 1e-12 and (start_i < best_cross[0] if best_cross else True)):
                                            best_new_max = new_dist
                                            best_cross = (i, j, start_i, start_j, len_i, len_j, new_i, new_j)
                if best_cross and best_new_max < current_max - 1e-12:
                    i, j, start_i, start_j, len_i, len_j, new_i, new_j = best_cross
                    routes[i] = new_i
                    routes[j] = new_j
                    current_max = best_new_max
                    report_best_vrp(routes)
                    improved = True

            if improved:
                improved_outer = True
                break  # restart neighborhood cycle
        if not improved_outer:
            break

    # Final intral route 2-opt (ensure all routes locally optimal)
    for r_idx in range(truck_count):
        improved = True
        while improved:
            improved = False
            route = routes[r_idx]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    a = route[i-1]
                    b = route[i]
                    c = route[j]
                    d = route[j+1]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    newd = distance_matrix[a][c] + distance_matrix[b][d]
                    if newd < old - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break
    return routes