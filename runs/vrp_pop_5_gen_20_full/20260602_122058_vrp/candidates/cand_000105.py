import numpy as np
import math
import random
from collections import defaultdict

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
        maxd = 0.0
        for r in routes:
            d = route_dist(r)
            if d > maxd:
                maxd = d
        return maxd

    def copy_routes(routes):
        return [r[:] for r in routes]

    def construct_initial():
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
        routes = [[0, s, 0] for s in seeds]
        remaining = [c for c in customers if c not in seeds]
        remaining.sort(key=lambda c: -distance_matrix[0, c])
        for cust in remaining:
            best_new_max = float('inf')
            best_route_idx = -1
            best_pos = -1
            for idx, route in enumerate(routes):
                best_delta = float('inf')
                best_pos_local = -1
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nex = route[pos]
                    delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                    if delta < best_delta:
                        best_delta = delta
                        best_pos_local = pos
                current_route_dist = route_dist(route)
                new_route_dist = current_route_dist + best_delta
                other_max = 0.0
                for j, r in enumerate(routes):
                    if j == idx:
                        continue
                    other_max = max(other_max, route_dist(r))
                new_max = max(other_max, new_route_dist)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_route_idx = idx
                    best_pos = best_pos_local
                elif new_max == best_new_max:
                    if idx < best_route_idx:
                        best_route_idx = idx
                        best_pos = best_pos_local
            routes[best_route_idx].insert(best_pos, cust)
            report_best_vrp(routes)
        return routes

    def regret_insert(routes, cust_list):
        remaining = cust_list[:]
        while remaining:
            best_cust = None
            best_regret = -1
            best_route_idx = -1
            best_pos = -1
            best_delta_val = float('inf')
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
                if len(deltas) >= 2:
                    regret = deltas[1][0] - deltas[0][0]
                else:
                    regret = 0
                if regret > best_regret or (regret == best_regret and deltas[0][0] < best_delta_val):
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = deltas[0][1]
                    best_pos = deltas[0][2]
                    best_delta_val = deltas[0][0]
                elif regret == best_regret and deltas[0][0] == best_delta_val and cust < best_cust:
                    best_cust = cust
            if best_cust is None:
                best_cust = remaining[0]
                best_route_idx = 0
                best_pos = 1
            routes[best_route_idx].insert(best_pos, best_cust)
            remaining.remove(best_cust)
        return routes

    def perturb_on_critical(routes, remove_frac=0.2):
        route_dists = [route_dist(r) for r in routes]
        max_dist = max(route_dists)
        critical_indices = [i for i, d in enumerate(route_dists) if d >= 0.9 * max_dist]
        if not critical_indices:
            critical_indices = [route_dists.index(max_dist)]
        all_custs = []
        for idx in critical_indices:
            route = routes[idx]
            for node in route[1:-1]:
                all_custs.append(node)
        random.shuffle(all_custs)
        remove_count = max(1, int(remove_frac * (n-1)))
        to_remove = all_custs[:remove_count]
        for cust in to_remove:
            for route in routes:
                if cust in route:
                    route.remove(cust)
                    break
        routes = regret_insert(routes, to_remove)
        return routes

    def tabu_search(initial_routes):
        current_routes = copy_routes(initial_routes)
        best_routes = copy_routes(initial_routes)
        best_max = compute_max_dist(best_routes)
        current_max = best_max

        max_iter = 200  # fixed to limit runtime
        tabu_tenure = 5
        tabu = {}
        iteration = 0
        no_improve_iter = 0

        while iteration < max_iter:
            iteration += 1
            no_improve_iter += 1

            best_move = None
            best_new_max = float('inf')
            best_move_type = None
            best_move_params = None

            # Relocate moves
            for i in range(truck_count):
                route_i = current_routes[i]
                if len(route_i) <= 2:
                    continue
                for pos in range(1, len(route_i)-1):
                    cust = route_i[pos]
                    for j in range(truck_count):
                        if j == i:
                            continue
                        route_j = current_routes[j]
                        best_delta = float('inf')
                        best_ins_pos = -1
                        for ins_pos in range(1, len(route_j)):
                            prev = route_j[ins_pos-1]
                            nex = route_j[ins_pos]
                            delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                            if delta < best_delta:
                                best_delta = delta
                                best_ins_pos = ins_pos
                        new_route_i = route_i[:pos] + route_i[pos+1:]
                        new_route_j = route_j[:best_ins_pos] + [cust] + route_j[best_ins_pos:]
                        dist_i = route_dist(new_route_i)
                        dist_j = route_dist(new_route_j)
                        other_max = 0.0
                        for k, r in enumerate(current_routes):
                            if k == i or k == j:
                                continue
                            other_max = max(other_max, route_dist(r))
                        new_max = max(other_max, dist_i, dist_j)

                        move_key = ('relocate', cust, i, j)
                        is_tabu = False
                        if move_key in tabu and tabu[move_key] > iteration:
                            is_tabu = True
                        if is_tabu and new_max >= best_max:
                            continue
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('relocate', i, pos, j, best_ins_pos)
                            best_move_type = 'relocate'
                            best_move_params = (i, pos, j, best_ins_pos)

            # 2-opt intra-route
            for i in range(truck_count):
                route_i = current_routes[i]
                if len(route_i) <= 3:
                    continue
                for a in range(1, len(route_i)-2):
                    for b in range(a+1, len(route_i)-1):
                        new_route_i = route_i[:a] + route_i[a:b+1][::-1] + route_i[b+1:]
                        dist_i = route_dist(new_route_i)
                        other_max = 0.0
                        for k, r in enumerate(current_routes):
                            if k == i:
                                continue
                            other_max = max(other_max, route_dist(r))
                        new_max = max(other_max, dist_i)

                        move_key = ('2opt', i, a, b)
                        is_tabu = False
                        if move_key in tabu and tabu[move_key] > iteration:
                            is_tabu = True
                        if is_tabu and new_max >= best_max:
                            continue
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('2opt', i, a, b)
                            best_move_type = '2opt'
                            best_move_params = (i, a, b)

            if best_move is None:
                break

            if best_move_type == 'relocate':
                _, i, pos, j, ins_pos = best_move
                cust = current_routes[i][pos]
                current_routes[i] = current_routes[i][:pos] + current_routes[i][pos+1:]
                current_routes[j] = current_routes[j][:ins_pos] + [cust] + current_routes[j][ins_pos:]
                tabu[('relocate', cust, j, i)] = iteration + tabu_tenure
            elif best_move_type == '2opt':
                _, i, a, b = best_move
                current_routes[i] = current_routes[i][:a] + current_routes[i][a:b+1][::-1] + current_routes[i][b+1:]
                tabu[('2opt', i, a, b)] = iteration + tabu_tenure

            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = copy_routes(current_routes)
                report_best_vrp(best_routes)
                no_improve_iter = 0

            if no_improve_iter >= 20:
                # perturb
                current_routes = perturb_on_critical(best_routes)
                current_max = compute_max_dist(current_routes)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = copy_routes(current_routes)
                    report_best_vrp(best_routes)
                tabu.clear()
                no_improve_iter = 0
                report_best_vrp(current_routes)

        return best_routes

    random.seed(12345)
    initial = construct_initial()
    best = tabu_search(initial)
    while len(best) < truck_count:
        best.append([0, 0])
    report_best_vrp(best)
    return best