import numpy as np
import random

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

    def construct_initial():
        routes = [[0, 0] for _ in range(truck_count)]
        order = sorted(customers, key=lambda c: -distance_matrix[0, c])
        for cust in order:
            best_new_max = float('inf')
            best_route_idx = -1
            best_pos = -1
            for idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_dist(route) + delta
                    other_max = 0.0
                    for j, r in enumerate(routes):
                        if j == idx:
                            continue
                        other_max = max(other_max, route_dist(r))
                    new_max = max(other_max, new_route_dist)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_route_idx = idx
                        best_pos = pos
            routes[best_route_idx].insert(best_pos, cust)
        return routes

    def local_search(routes):
        max_iter = min(50, n * truck_count)
        for _ in range(max_iter):
            improved = False
            current_max = max_dist(routes)
            # Relocate
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos in range(1, len(route_i) - 1):
                    cust = route_i[pos]
                    for j in range(truck_count):
                        if j == i:
                            continue
                        route_j = routes[j]
                        for ins_pos in range(1, len(route_j)):
                            new_route_i = route_i[:pos] + route_i[pos+1:]
                            new_route_j = route_j[:ins_pos] + [cust] + route_j[ins_pos:]
                            dist_i = route_dist(new_route_i)
                            dist_j = route_dist(new_route_j)
                            other_max = 0.0
                            for k, r in enumerate(routes):
                                if k == i or k == j:
                                    continue
                                other_max = max(other_max, route_dist(r))
                            new_max = max(other_max, dist_i, dist_j)
                            if new_max < current_max:
                                routes[i] = new_route_i
                                routes[j] = new_route_j
                                improved = True
                                current_max = new_max
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt intra-route
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 3:
                    continue
                for a in range(1, len(route_i) - 2):
                    for b in range(a + 1, len(route_i) - 1):
                        new_route_i = route_i[:a] + route_i[a:b+1][::-1] + route_i[b+1:]
                        dist_i = route_dist(new_route_i)
                        other_max = 0.0
                        for j, r in enumerate(routes):
                            if j == i:
                                continue
                            other_max = max(other_max, route_dist(r))
                        new_max = max(other_max, dist_i)
                        if new_max < current_max:
                            routes[i] = new_route_i
                            improved = True
                            current_max = new_max
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    best_routes = None
    best_max = float('inf')
    # Initial solution
    routes = construct_initial()
    routes = local_search(routes)
    curr_max = max_dist(routes)
    if curr_max < best_max:
        best_max = curr_max
        best_routes = [r[:] for r in routes]
        report_best_vrp(best_routes)

    # Restart loop
    for restart in range(10):
        random.seed(restart + 999)
        # Remove a random subset of customers
        all_customers = list(customers)
        random.shuffle(all_customers)
        remove_count = max(1, len(all_customers) // 5)
        to_remove = set(all_customers[:remove_count])
        # Construct new routes by removing those customers
        new_routes = []
        for r in routes:
            new_route = [0]
            for node in r[1:-1]:
                if node not in to_remove:
                    new_route.append(node)
            new_route.append(0)
            new_routes.append(new_route)
        # Reinsert removed customers in the original random order
        remaining = [c for c in all_customers if c in to_remove]
        for cust in remaining:
            best_new_max = float('inf')
            best_route_idx = -1
            best_pos = -1
            for idx, route in enumerate(new_routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_route_dist = route_dist(route) + delta
                    other_max = 0.0
                    for j, r in enumerate(new_routes):
                        if j == idx:
                            continue
                        other_max = max(other_max, route_dist(r))
                    new_max = max(other_max, new_route_dist)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_route_idx = idx
                        best_pos = pos
            new_routes[best_route_idx].insert(best_pos, cust)
        # Local search on perturbed solution
        new_routes = local_search(new_routes)
        new_max_d = max_dist(new_routes)
        if new_max_d < best_max:
            best_max = new_max_d
            best_routes = [r[:] for r in new_routes]
            report_best_vrp(best_routes)
        # Accept perturbed as current for next restart
        routes = new_routes

    if best_routes is None:
        best_routes = routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    report_best_vrp(best_routes)
    return best_routes