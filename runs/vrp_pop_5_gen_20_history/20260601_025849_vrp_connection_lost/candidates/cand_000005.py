import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # handle degenerate case
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    # if truck_count >= n-1, assign each customer its own route
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        # pad with empty routes
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes
    # initial solution: each route starts at depot
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0] * truck_count
    # assign customers in descending order of distance from depot
    sorted_customers = sorted(customers, key=lambda c: distance_matrix[0, c], reverse=True)
    for cust in sorted_customers:
        best_route = None
        best_pos = None
        best_increase = float('inf')
        for r_idx, route in enumerate(routes):
            # try inserting at every position
            for pos in range(1, len(route)):
                prev = route[pos-1]
                next_node = route[pos]
                inc = distance_matrix[prev, cust] + distance_matrix[cust, next_node] - distance_matrix[prev, next_node]
                if inc < best_increase:
                    best_increase = inc
                    best_route = r_idx
                    best_pos = pos
        route = routes[best_route]
        route.insert(best_pos, cust)
        route_distances[best_route] += best_increase
    # define helper functions
    def calc_route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    def calc_all_distances(routes):
        return [calc_route_dist(r) for r in routes]
    def best_max(routes):
        return max(calc_all_distances(routes))
    # initial best
    best_routes = [r[:] for r in routes]
    best_max_val = best_max(routes)
    # report initial
    report_best_vrp(best_routes)
    # local search parameters
    max_iter = 100 * n
    iter_count = 0
    improved = True
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        # relocate: move one customer to another route
        for i in range(truck_count):
            for j in range(len(routes[i])-1):  # skip depot positions
                cust = routes[i][j]
                if cust == 0:
                    continue
                # remove customer from its route
                new_route_i = routes[i][:j] + routes[i][j+1:]
                dist_i = calc_route_dist(new_route_i)
                for k in range(truck_count):
                    if k == i:
                        continue
                    # try inserting into route k at every position
                    for pos in range(1, len(routes[k])+1):
                        new_route_k = routes[k][:pos] + [cust] + routes[k][pos:]
                        dist_k = calc_route_dist(new_route_k)
                        new_max = max(dist_i, dist_k, max(route_distances[:i]+route_distances[i+1:]))  # rough
                        current_max = best_max(routes)
                        if new_max < current_max - 1e-9:
                            # accept move
                            routes[i] = new_route_i
                            routes[k] = new_route_k
                            route_distances = calc_all_distances(routes)
                            improved = True
                            if best_max(routes) < best_max_val - 1e-9:
                                best_routes = [r[:] for r in routes]
                                best_max_val = best_max(routes)
                                report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # swap: exchange customers between two routes
        for i in range(truck_count):
            for j in range(1, len(routes[i])-1):
                cust_i = routes[i][j]
                for k in range(i+1, truck_count):
                    for l in range(1, len(routes[k])-1):
                        cust_j = routes[k][l]
                        # swap
                        new_routes = [r[:] for r in routes]
                        new_routes[i][j] = cust_j
                        new_routes[k][l] = cust_i
                        new_max = best_max(new_routes)
                        if new_max < best_max_val - 1e-9:
                            routes = new_routes
                            route_distances = calc_all_distances(routes)
                            improved = True
                            best_routes = [r[:] for r in routes]
                            best_max_val = best_max(routes)
                            report_best_vrp(best_routes)
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
            for a in range(1, len(route)-2):
                for b in range(a+1, len(route)-1):
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    new_dist = calc_route_dist(new_route)
                    if new_dist < route_distances[i] - 1e-9:
                        routes[i] = new_route
                        route_distances[i] = new_dist
                        improved = True
                        if best_max(routes) < best_max_val - 1e-9:
                            best_routes = [r[:] for r in routes]
                            best_max_val = best_max(routes)
                            report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # cross-route 2-opt (2-opt*)
        for i in range(truck_count):
            for j in range(i+1, truck_count):
                route_i = routes[i]
                route_j = routes[j]
                if len(route_i) <= 2 or len(route_j) <= 2:
                    continue
                for p in range(1, len(route_i)-1):
                    for q in range(1, len(route_j)-1):
                        # swap tails
                        new_i = route_i[:p] + route_j[q:]
                        new_j = route_j[:q] + route_i[p:]
                        new_dist_i = calc_route_dist(new_i)
                        new_dist_j = calc_route_dist(new_j)
                        # check if feasible: routes must start and end at depot, which they do
                        old_max = max(route_distances[i], route_distances[j])
                        new_max = max(new_dist_i, new_dist_j)
                        overall_max = max(route_distances[:i]+route_distances[i+1:j]+route_distances[j+1:]+[new_dist_i, new_dist_j])
                        if overall_max < best_max_val - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
                            route_distances[i] = new_dist_i
                            route_distances[j] = new_dist_j
                            improved = True
                            best_routes = [r[:] for r in routes]
                            best_max_val = best_max(routes)
                            report_best_vrp(best_routes)
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    # final best
    return best_routes