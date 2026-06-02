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

    def regret_insertion(remaining, routes, regret_k=2):
        rem = list(remaining)
        while rem:
            best_cust = None
            best_route_idx = -1
            best_pos = -1
            best_regret = -float('inf')
            for cust in rem:
                deltas = []
                for idx, route in enumerate(routes):
                    best_delta = float('inf')
                    best_local_pos = -1
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nex = route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                        if delta < best_delta:
                            best_delta = delta
                            best_local_pos = pos
                    deltas.append((best_delta, best_local_pos, idx))
                deltas.sort(key=lambda x: x[0])
                if len(deltas) >= 2:
                    regret = deltas[1][0] - deltas[0][0]
                else:
                    regret = deltas[0][0]
                if regret > best_regret or (regret == best_regret and best_cust is not None and cust < best_cust):
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = deltas[0][2]
                    best_pos = deltas[0][1]
            routes[best_route_idx].insert(best_pos, best_cust)
            rem.remove(best_cust)
            report_best_vrp(routes)
        return routes

    def construct_initial(seeds):
        routes = [[0, s, 0] for s in seeds]
        remaining = [c for c in customers if c not in seeds]
        remaining.sort(key=lambda c: -distance_matrix[0, c])
        return regret_insertion(remaining, routes, regret_k=2)

    # Seed selection: farthest from depot and each other
    seeds = []
    first_seed = max(range(1, n), key=lambda i: (distance_matrix[0, i], -i))
    seeds.append(first_seed)
    for _ in range(1, truck_count):
        best_min_dist = -1.0
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

    best_routes = None
    best_max = float('inf')

    def best_improvement_local_search(routes):
        improved = True
        max_iter = n * truck_count  # finite bound
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            # Inter-node relocation (best improvement)
            best_move = None
            best_new_routes = None
            best_new_max = float('inf')
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust = route_i[pos_i]
                    new_route_i = route_i[:pos_i] + route_i[pos_i+1:]
                    for j in range(truck_count):
                        if i == j:
                            continue
                        route_j = routes[j]
                        for pos_j in range(1, len(route_j)):
                            new_route_j = route_j[:pos_j] + [cust] + route_j[pos_j:]
                            candidate_routes = routes[:]
                            candidate_routes[i] = new_route_i
                            candidate_routes[j] = new_route_j
                            cand_max = max_dist(candidate_routes)
                            if cand_max < best_new_max:
                                best_new_max = cand_max
                                best_move = ('relocate', i, pos_i, cust, j, pos_j)
                                best_new_routes = [r[:] for r in candidate_routes]
            # Inter-node swap
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust_i = route_i[pos_i]
                    for j in range(i+1, truck_count):
                        route_j = routes[j]
                        if len(route_j) <= 2:
                            continue
                        for pos_j in range(1, len(route_j)-1):
                            cust_j = route_j[pos_j]
                            new_route_i = route_i[:pos_i] + [cust_j] + route_i[pos_i+1:]
                            new_route_j = route_j[:pos_j] + [cust_i] + route_j[pos_j+1:]
                            candidate_routes = routes[:]
                            candidate_routes[i] = new_route_i
                            candidate_routes[j] = new_route_j
                            cand_max = max_dist(candidate_routes)
                            if cand_max < best_new_max:
                                best_new_max = cand_max
                                best_move = ('swap', i, pos_i, j, pos_j)
                                best_new_routes = [r[:] for r in candidate_routes]
            # Intra-route 2-opt
            for i in range(truck_count):
                route = routes[i]
                if len(route) <= 3:
                    continue
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        candidate_routes = routes[:]
                        candidate_routes[i] = new_route
                        cand_max = max_dist(candidate_routes)
                        if cand_max < best_new_max:
                            best_new_max = cand_max
                            best_move = ('2opt', i, a, b)
                            best_new_routes = [r[:] for r in candidate_routes]
            if best_new_max < max_dist(routes) - 1e-12:
                routes = best_new_routes
                improved = True
                report_best_vrp(routes)
        return routes

    # Initial solution
    routes = construct_initial(seeds)
    routes = best_improvement_local_search(routes)
    best_max = max_dist(routes)
    best_routes = [r[:] for r in routes]

    # Restarts
    max_restarts = 5
    for restart in range(max_restarts):
        # Remove up to 20% of customers
        all_customers_list = list(range(1, n))
        remove_count = max(1, n // 5)
        to_remove = set(random.sample(all_customers_list, remove_count))
        new_routes = []
        for r in best_routes:
            new_route = [0]
            for node in r[1:-1]:
                if node not in to_remove:
                    new_route.append(node)
            new_route.append(0)
            new_routes.append(new_route)
        remaining = list(to_remove)
        new_routes = regret_insertion(remaining, new_routes)
        new_routes = best_improvement_local_search(new_routes)
        new_max = max_dist(new_routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [r[:] for r in new_routes]
            report_best_vrp(best_routes)

    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    report_best_vrp(best_routes)
    return best_routes