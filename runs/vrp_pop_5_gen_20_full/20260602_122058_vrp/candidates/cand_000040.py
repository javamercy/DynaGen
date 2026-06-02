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

    def compute_max_dist(routes):
        return max(route_dist(r) for r in routes)

    def copy_routes(routes):
        return [r[:] for r in routes]

    # Construction: farthest-first seeds + greedy insertion
    def construct_initial():
        seeds = []
        # first seed: farthest from depot
        first_seed = max(range(1, n), key=lambda i: (distance_matrix[0, i], -i))
        seeds.append(first_seed)
        for _ in range(1, truck_count):
            best_min_dist = -1
            best_node = None
            for node in range(1, n):
                if node in seeds:
                    continue
                min_dist = min(distance_matrix[node, s] for s in seeds)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_node = node
                elif min_dist == best_min_dist and (best_node is None or node < best_node):
                    # tie break by smaller index
                    best_node = node
            if best_node is None:
                break
            seeds.append(best_node)
        # initialize routes with seeds
        routes = [[0, s, 0] for s in seeds]
        remaining = [c for c in customers if c not in seeds]
        # greedy insertion (minimize distance increase)
        for cust in remaining:
            best_delta = float('inf')
            best_route_idx = -1
            best_pos = -1
            for idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nex = route[pos]
                    delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                    if delta < best_delta:
                        best_delta = delta
                        best_route_idx = idx
                        best_pos = pos
            routes[best_route_idx].insert(best_pos, cust)
            report_best_vrp(routes)
        return routes

    # Local search: first-improvement relocate and swap
    def local_search(routes):
        improved = True
        max_iter = min(20, n * truck_count)  # finite bound
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
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
                            # compute new max distance
                            other_max = 0.0
                            for k, r in enumerate(routes):
                                if k == i or k == j:
                                    continue
                                other_max = max(other_max, route_dist(r))
                            new_max = max(other_max, route_dist(new_route_i), route_dist(new_route_j))
                            current_max = compute_max_dist(routes)
                            if new_max < current_max:
                                routes[i] = new_route_i
                                routes[j] = new_route_j
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Swap
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i) - 1):
                    cust_i = route_i[pos_i]
                    for j in range(i+1, truck_count):
                        route_j = routes[j]
                        if len(route_j) <= 2:
                            continue
                        for pos_j in range(1, len(route_j) - 1):
                            cust_j = route_j[pos_j]
                            new_route_i = route_i[:]
                            new_route_i[pos_i] = cust_j
                            new_route_j = route_j[:]
                            new_route_j[pos_j] = cust_i
                            other_max = 0.0
                            for k, r in enumerate(routes):
                                if k == i or k == j:
                                    continue
                                other_max = max(other_max, route_dist(r))
                            new_max = max(other_max, route_dist(new_route_i), route_dist(new_route_j))
                            current_max = compute_max_dist(routes)
                            if new_max < current_max:
                                routes[i] = new_route_i
                                routes[j] = new_route_j
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # Regret-2 insertion
    def regret_insert(routes, cust_list):
        remaining = cust_list[:]
        while remaining:
            best_cust = None
            best_regret = -1
            best_route_idx = -1
            best_pos = -1
            best_delta = float('inf')
            for cust in remaining:
                deltas = []
                for idx, route in enumerate(routes):
                    min_delta = float('inf')
                    best_pos_local = -1
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nex = route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                        if delta < min_delta:
                            min_delta = delta
                            best_pos_local = pos
                    deltas.append((min_delta, idx, best_pos_local))
                deltas.sort(key=lambda x: x[0])
                regret = deltas[1][0] - deltas[0][0] if len(deltas) >= 2 else 0
                if regret > best_regret or (regret == best_regret and deltas[0][0] < best_delta):
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = deltas[0][1]
                    best_pos = deltas[0][2]
                    best_delta = deltas[0][0]
                elif regret == best_regret and deltas[0][0] == best_delta and cust < best_cust:
                    best_cust = cust
            if best_cust is None:
                break
            routes[best_route_idx].insert(best_pos, best_cust)
            remaining.remove(best_cust)
        return routes

    # Perturbation: remove 20% customers randomly and reinsert via regret-2
    def perturb(routes):
        custom = copy_routes(routes)
        all_custs = list(range(1, n))
        random.shuffle(all_custs)
        remove_count = max(1, int(0.2 * (n - 1)))
        to_remove = all_custs[:remove_count]
        removed_custs = []
        for cust in to_remove:
            for route in custom:
                if cust in route:
                    route.remove(cust)
                    removed_custs.append(cust)
                    break
        custom = regret_insert(custom, removed_custs)
        return custom

    # Main
    random.seed(42)
    best_routes = construct_initial()
    best_routes = local_search(best_routes)
    best_max = compute_max_dist(best_routes)
    report_best_vrp(best_routes)

    for restart in range(10):
        new_routes = perturb(best_routes)
        new_routes = local_search(new_routes)
        new_max = compute_max_dist(new_routes)
        if new_max < best_max:
            best_routes = new_routes
            best_max = new_max
            report_best_vrp(best_routes)
    return best_routes