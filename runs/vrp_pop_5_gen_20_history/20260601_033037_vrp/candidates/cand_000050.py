import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = len(distance_matrix)
    customers = list(range(1, n))
    if truck_count >= n:
        routes = [[0, c, 0] for c in customers]
        routes += [[0, 0]] * (truck_count - len(customers))
        return routes

    # farthest insertion construction
    routes = [[0, 0] for _ in range(truck_count)]
    # first customer for each route: farthest from depot
    seeds = sorted(customers, key=lambda x: distance_matrix[0][x], reverse=True)[:truck_count]
    for i, seed in enumerate(seeds):
        routes[i] = [0, seed, 0]
    unvisited = [c for c in customers if c not in seeds]
    while unvisited:
        # find customer with max distance to any route
        best_cust = None
        best_dist = -1
        for c in unvisited:
            # distance to nearest route
            min_dist = float('inf')
            for route in routes:
                for node in route:
                    if node == 0:
                        continue
                    d = distance_matrix[c][node]
                    if d < min_dist:
                        min_dist = d
            if min_dist > best_dist:
                best_dist = min_dist
                best_cust = c
        # insert best_cust into best position (cheapest insertion cost)
        best_route_idx = None
        best_insert_cost = float('inf')
        best_pos = None
        for i, route in enumerate(routes):
            cost_without = sum(distance_matrix[route[j]][route[j+1]] for j in range(len(route)-1))
            for pos in range(1, len(route)):
                new_route = route[:pos] + [best_cust] + route[pos:]
                cost_with = sum(distance_matrix[new_route[j]][new_route[j+1]] for j in range(len(new_route)-1))
                delta = cost_with - cost_without
                if delta < best_insert_cost:
                    best_insert_cost = delta
                    best_route_idx = i
                    best_pos = pos
        routes[best_route_idx] = routes[best_route_idx][:best_pos] + [best_cust] + routes[best_route_idx][best_pos:]
        unvisited.remove(best_cust)
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in routes)

    def route_distance(route):
        if len(route) <= 1:
            return 0
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

    def report_best(routes):
        nonlocal best_routes, best_max
        max_d = max(route_distance(r) for r in routes)
        if max_d < best_max:
            best_max = max_d
            best_routes = [list(r) for r in routes]

    # improvement: inter-route relocate and intra-route 2-opt
    max_iter = n * truck_count
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
                        if new_max < best_max:
                            routes = new_routes
                            report_best(routes)
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
            best_route = list(route)
            best_dist = route_distance(route)
            for a in range(1, len(route)-2):
                for b in range(a+1, len(route)-1):
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    if new_route[0] != 0 or new_route[-1] != 0:
                        continue
                    d = route_distance(new_route)
                    if d < best_dist:
                        best_dist = d
                        best_route = new_route
                        improved = True
            if improved:
                routes[i] = best_route
                report_best(routes)
        if not improved:
            break
    return best_routes