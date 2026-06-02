import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    # Cheapest insertion (regret-1) for given customer and list of routes
    def cheapest_insertion(remaining, routes):
        rem = list(remaining)
        while rem:
            best_cust = None
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            for cust in rem:
                for idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nex = route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                        if delta < best_cost or (delta == best_cost and (cust < best_cust or (cust == best_cust and (idx < best_route_idx or (idx == best_route_idx and pos < best_pos))))):
                            best_cost = delta
                            best_cust = cust
                            best_route_idx = idx
                            best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            rem.remove(best_cust)
            report_best_vrp(routes)
        return routes

    # Seed selection: farthest from depot and each other
    seeds = []
    first_seed = max(range(1, n), key=lambda i: (distance_matrix[0, i], -i))
    seeds.append(first_seed)
    for _ in range(1, truck_count):
        best_min_dist = -1
        best_node = None
        for node in range(1, n):
            if node in seeds:
                continue
            min_dist = min(distance_matrix[node, s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and (best_node is None or node < best_node)):
                best_min_dist = min_dist
                best_node = node
        if best_node is None:
            break
        seeds.append(best_node)

    # Initial routes
    routes = [[0, s, 0] for s in seeds]
    remaining = [c for c in customers if c not in seeds]
    remaining.sort(key=lambda c: -distance_matrix[0, c])
    routes = cheapest_insertion(remaining, routes)
    best_routes = [r[:] for r in routes]
    best_max = max_dist(best_routes)

    # Steepest descent local search
    def local_search(routes):
        improved = True
        max_iter = n * truck_count
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            current_routes = [r[:] for r in routes]
            current_max = max_dist(current_routes)
            best_move = None
            best_new_routes = None
            best_new_max = current_max

            # Relocate moves
            for i in range(truck_count):
                for pos in range(1, len(current_routes[i])-1):
                    cust = current_routes[i][pos]
                    for j in range(truck_count):
                        if j == i:
                            continue
                        for ins_pos in range(1, len(current_routes[j])):
                            new_routes = [r[:] for r in current_routes]
                            new_routes[i].pop(pos)
                            new_routes[j].insert(ins_pos, cust)
                            new_max = max_dist(new_routes)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('relocate', i, pos, j, ins_pos)
                                best_new_routes = new_routes

            # Swap moves
            for i in range(truck_count):
                for pos_i in range(1, len(current_routes[i])-1):
                    cust_i = current_routes[i][pos_i]
                    for j in range(i+1, truck_count):
                        for pos_j in range(1, len(current_routes[j])-1):
                            cust_j = current_routes[j][pos_j]
                            new_routes = [r[:] for r in current_routes]
                            new_routes[i][pos_i] = cust_j
                            new_routes[j][pos_j] = cust_i
                            new_max = max_dist(new_routes)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('swap', i, pos_i, j, pos_j)
                                best_new_routes = new_routes

            # 2-opt intra-route moves
            for i in range(truck_count):
                if len(current_routes[i]) <= 3:
                    continue
                for a in range(1, len(current_routes[i])-2):
                    for b in range(a+1, len(current_routes[i])-1):
                        new_route = current_routes[i][:a] + current_routes[i][a:b+1][::-1] + current_routes[i][b+1:]
                        new_routes = [r[:] for r in current_routes]
                        new_routes[i] = new_route
                        new_max = max_dist(new_routes)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('2opt', i, a, b)
                            best_new_routes = new_routes

            if best_move is not None:
                routes = best_new_routes
                best_max = best_new_max
                improved = True
                report_best_vrp(routes)
            iteration += 1
        return routes, best_max

    routes, best_max = local_search(routes)
    best_routes = [r[:] for r in routes]
    best_max = best_max

    # Perturbation and restart
    max_restarts = 5
    random.seed(12345)
    for restart in range(max_restarts):
        all_customers = list(range(1, n))
        remove_count = max(1, n // 10)
        to_remove = set(random.sample(all_customers, remove_count))
        new_routes = []
        for r in best_routes:
            new_route = [0]
            for node in r[1:-1]:
                if node not in to_remove:
                    new_route.append(node)
            new_route.append(0)
            new_routes.append(new_route)
        remaining = list(to_remove)
        new_routes = cheapest_insertion(remaining, new_routes)
        new_routes, new_max = local_search(new_routes)
        if new_max < best_max:
            best_max = new_max
            best_routes = [r[:] for r in new_routes]
            report_best_vrp(best_routes)

    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    report_best_vrp(best_routes)
    return best_routes