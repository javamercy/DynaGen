import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = len(distance_matrix)
    customers = list(range(1, n))
    best_routes = None
    best_max_dist = float('inf')

    def route_distance(route):
        if len(route) <= 1:
            return 0
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

    def compute_routes_and_max(clusters):
        routes = []
        for cl in clusters:
            if not cl:
                routes.append([0, 0])
            else:
                route = [0]
                unvisited = list(cl)
                current = 0
                while unvisited:
                    nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
                    route.append(nearest)
                    unvisited.remove(nearest)
                    current = nearest
                route.append(0)
                routes.append(route)
        return routes, max(route_distance(r) for r in routes)

    def report_best_vrp(routes):
        nonlocal best_routes, best_max_dist
        max_d = max(route_distance(r) for r in routes)
        if max_d < best_max_dist:
            best_max_dist = max_d
            best_routes = [list(r) for r in routes]

    if truck_count >= n:
        clusters = [[c] for c in customers] + [[] for _ in range(truck_count - len(customers))]
        routes, _ = compute_routes_and_max(clusters)
        report_best_vrp(routes)
        return best_routes

    # farthest-first seed selection
    seeds = []
    seed0 = max(customers, key=lambda x: distance_matrix[0][x])
    seeds.append(seed0)
    while len(seeds) < truck_count:
        best_customer = None
        best_min_dist = -1
        for c in customers:
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c][s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and c < (best_customer if best_customer is not None else float('inf'))):
                best_min_dist = min_dist
                best_customer = c
        if best_customer is not None:
            seeds.append(best_customer)
        else:
            break
    clusters = [[] for _ in range(truck_count)]
    for c in customers:
        if c in seeds:
            continue
        min_dist = float('inf')
        best_idx = 0
        for i, s in enumerate(seeds):
            d = distance_matrix[c][s]
            if d < min_dist or (d == min_dist and i < best_idx):
                min_dist = d
                best_idx = i
        clusters[best_idx].append(c)
    for i, s in enumerate(seeds):
        clusters[i].append(s)
    routes, _ = compute_routes_and_max(clusters)
    report_best_vrp(routes)

    # improvement: local search with relocate, swap, and intra-route Or-opt
    max_iter = min(n * truck_count, 200)
    for iteration in range(max_iter):
        improved = False
        # inter-route relocate
        for i in range(truck_count):
            route_i = routes[i]
            if len(route_i) <= 2:
                continue
            for cust in route_i[1:-1]:
                for j in range(truck_count):
                    if i == j:
                        continue
                    route_j = routes[j]
                    for pos in range(1, len(route_j)):
                        new_routes = [list(r) for r in routes]
                        new_routes[i].remove(cust)
                        new_routes[j].insert(pos, cust)
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max_dist:
                            routes = new_routes
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # inter-route swap
        for i in range(truck_count):
            route_i = routes[i]
            if len(route_i) <= 2:
                continue
            for cust_i in route_i[1:-1]:
                for j in range(i+1, truck_count):
                    route_j = routes[j]
                    if len(route_j) <= 2:
                        continue
                    for cust_j in route_j[1:-1]:
                        new_routes = [list(r) for r in routes]
                        idx_i = new_routes[i].index(cust_i)
                        idx_j = new_routes[j].index(cust_j)
                        new_routes[i][idx_i], new_routes[j][idx_j] = cust_j, cust_i
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max_dist:
                            routes = new_routes
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # intra-route Or-opt
        for i in range(truck_count):
            route = routes[i]
            if len(route) <= 3:
                continue
            best_route = list(route)
            best_dist = route_distance(route)
            for seg_len in range(1, min(4, len(route)-1)):  # segment length 1 to 3
                for start in range(1, len(route)-seg_len):
                    end = start + seg_len - 1
                    segment = route[start:end+1]
                    # remove segment
                    remaining = route[:start] + route[end+1:]
                    # try inserting at all positions (including before first and after last)
                    for pos in range(1, len(remaining)):
                        new_route = remaining[:pos] + segment + remaining[pos:]
                        # ensure starts and ends with 0
                        if new_route[0] != 0 or new_route[-1] != 0:
                            continue
                        d = route_distance(new_route)
                        if d < best_dist:
                            best_dist = d
                            best_route = new_route
                            improved = True
            if improved:
                routes[i] = best_route
                report_best_vrp(routes)
        if not improved:
            break
    max_d = max(route_distance(r) for r in routes)
    if max_d < best_max_dist:
        best_routes = [list(r) for r in routes]
    return best_routes