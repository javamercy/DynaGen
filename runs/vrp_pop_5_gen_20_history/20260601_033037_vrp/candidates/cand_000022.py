import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = len(distance_matrix)
    customers = list(range(1, n))
    best_routes = None
    best_max_dist = float('inf')

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def compute_routes_and_max(clusters):
        routes = []
        for cl in clusters:
            if not cl:
                routes.append([0, 0])
            else:
                unvisited = list(cl)
                route = [0, 0]
                while unvisited:
                    best_customer = None
                    best_pos = None
                    best_cost = float('inf')
                    for c in unvisited:
                        for pos in range(1, len(route)):
                            delta = (distance_matrix[route[pos-1], c] +
                                     distance_matrix[c, route[pos]] -
                                     distance_matrix[route[pos-1], route[pos]])
                            if delta < best_cost or (delta == best_cost and (best_customer is None or c < best_customer)):
                                best_cost = delta
                                best_customer = c
                                best_pos = pos
                    route.insert(best_pos, best_customer)
                    unvisited.remove(best_customer)
                routes.append(route)
        return routes, max(route_distance(r) for r in routes)

    def report_best_vrp(routes):
        nonlocal best_routes, best_max_dist
        max_d = max(route_distance(r) for r in routes)
        if max_d < best_max_dist:
            best_max_dist = max_d
            best_routes = [list(r) for r in routes]

    # Special case: truck_count >= number of customers
    if truck_count >= n - 1:
        clusters = [[c] for c in customers] + [[] for _ in range(truck_count - (n - 1))]
        routes, _ = compute_routes_and_max(clusters)
        report_best_vrp(routes)
        return best_routes

    # Farthest-first seed selection
    seeds = []
    seed0 = max(customers, key=lambda x: distance_matrix[0, x])
    seeds.append(seed0)
    while len(seeds) < truck_count:
        best_customer = None
        best_min_dist = -1.0
        for c in customers:
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c, s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and best_customer is not None and c < best_customer):
                best_min_dist = min_dist
                best_customer = c
        if best_customer is not None:
            seeds.append(best_customer)
        else:
            break

    # Assign remaining customers to nearest seed
    clusters = [[] for _ in range(truck_count)]
    for c in customers:
        if c in seeds:
            continue
        min_dist = float('inf')
        best_idx = 0
        for i, s in enumerate(seeds):
            d = distance_matrix[c, s]
            if d < min_dist or (d == min_dist and i < best_idx):
                min_dist = d
                best_idx = i
        clusters[best_idx].append(c)
    for i, s in enumerate(seeds):
        clusters[i].append(s)

    # Build initial routes using cheapest insertion
    routes, _ = compute_routes_and_max(clusters)
    report_best_vrp(routes)

    # Local improvement: inter-route relocate, swap, and intra-route 2-opt
    max_iter = min(n * truck_count, 200)
    for _ in range(max_iter):
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
        # intra-route 2-opt
        for i in range(truck_count):
            route = routes[i]
            if len(route) <= 3:
                continue
            best_swap = None
            best_gain = 0.0
            for i1 in range(1, len(route)-2):
                for i2 in range(i1+1, len(route)-1):
                    new_route = route[:i1] + route[i1:i2+1][::-1] + route[i2+1:]
                    gain = route_distance(route) - route_distance(new_route)
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = new_route
            if best_gain > 0:
                routes[i] = best_swap
                improved = True
                report_best_vrp(routes)
        if not improved:
            break

    max_d = max(route_distance(r) for r in routes)
    if max_d < best_max_dist:
        best_routes = [list(r) for r in routes]
    return best_routes