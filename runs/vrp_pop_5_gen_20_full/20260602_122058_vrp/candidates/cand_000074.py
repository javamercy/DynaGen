import numpy as np
import math
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

    # Farthest insertion construction
    def construct_initial():
        seeds = []
        first = max(customers, key=lambda i: distance_matrix[0, i])
        seeds.append(first)
        for _ in range(1, truck_count):
            best_min = -1
            best_node = None
            for node in customers:
                if node in seeds:
                    continue
                min_dist = min(distance_matrix[node, s] for s in seeds)
                if min_dist > best_min:
                    best_min = min_dist
                    best_node = node
            if best_node is None:
                break
            seeds.append(best_node)
        routes = [[0, s, 0] for s in seeds]
        remaining = [c for c in customers if c not in seeds]
        remaining.sort(key=lambda c: -distance_matrix[0, c])
        for cust in remaining:
            best_new_max = float('inf')
            best_route_idx = 0
            best_pos = 1
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
                new_dist = route_dist(route) + best_delta
                other_max = 0.0
                for j, r in enumerate(routes):
                    if j == idx:
                        continue
                    other_max = max(other_max, route_dist(r))
                new_max = max(other_max, new_dist)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_route_idx = idx
                    best_pos = best_pos_local
            routes[best_route_idx].insert(best_pos, cust)
            report_best_vrp(routes)
        return routes

    def tabu_search(initial):
        current = copy_routes(initial)
        best = copy_routes(initial)
        best_max = compute_max_dist(best)
        current_max = best_max

        max_iter = 100 + n * truck_count // 2  # limit iterations
        base_tenure = max(1, int(math.sqrt(n)))
        tabu = {}
        iteration = 0
        no_improve = 0

        while iteration < max_iter:
            iteration += 1
            no_improve += 1

            best_move = None
            best_new_max = float('inf')
            best_move_key = None

            # Relocate moves
            for i in range(truck_count):
                route_i = current[i]
                if len(route_i) <= 2:
                    continue
                for pos in range(1, len(route_i)-1):
                    cust = route_i[pos]
                    for j in range(truck_count):
                        if j == i:
                            continue
                        route_j = current[j]
                        # best insertion position in route_j
                        best_delta = float('inf')
                        best_ins = -1
                        for ins_pos in range(1, len(route_j)):
                            prev = route_j[ins_pos-1]
                            nex = route_j[ins_pos]
                            delta = distance_matrix[prev, cust] + distance_matrix[cust, nex] - distance_matrix[prev, nex]
                            if delta < best_delta:
                                best_delta = delta
                                best_ins = ins_pos
                        new_i = route_i[:pos] + route_i[pos+1:]
                        new_j = route_j[:best_ins] + [cust] + route_j[best_ins:]
                        new_max = max(route_dist(new_i), route_dist(new_j), max(route_dist(r) for idx2, r in enumerate(current) if idx2 not in (i, j)))
                        move_key = ('reloc', cust, i, j)
                        is_tabu = move_key in tabu and tabu[move_key] > iteration
                        if is_tabu and new_max >= best_max:
                            continue
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = (i, pos, j, best_ins)
                            best_move_key = move_key

            # Swap moves
            for i in range(truck_count):
                route_i = current[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust_i = route_i[pos_i]
                    for j in range(i+1, truck_count):
                        route_j = current[j]
                        if len(route_j) <= 2:
                            continue
                        for pos_j in range(1, len(route_j)-1):
                            cust_j = route_j[pos_j]
                            new_i = route_i[:]
                            new_i[pos_i] = cust_j
                            new_j = route_j[:]
                            new_j[pos_j] = cust_i
                            new_max = max(route_dist(new_i), route_dist(new_j), max(route_dist(r) for idx2, r in enumerate(current) if idx2 not in (i, j)))
                            move_key = ('swap', cust_i, cust_j, i, j)
                            is_tabu = move_key in tabu and tabu[move_key] > iteration
                            if is_tabu and new_max >= best_max:
                                continue
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('swap', i, pos_i, j, pos_j)
                                best_move_key = move_key

            if best_move is None:
                break

            # Apply move
            if best_move[0] == 'swap':
                _, i, pos_i, j, pos_j = best_move
                cust_i = current[i][pos_i]
                cust_j = current[j][pos_j]
                current[i][pos_i] = cust_j
                current[j][pos_j] = cust_i
                tabu[('swap', cust_i, cust_j, i, j)] = iteration + base_tenure
                tabu[('swap', cust_j, cust_i, j, i)] = iteration + base_tenure
            else:
                i, pos, j, ins = best_move
                cust = current[i][pos]
                current[i] = current[i][:pos] + current[i][pos+1:]
                current[j] = current[j][:ins] + [cust] + current[j][ins:]
                tabu[('reloc', cust, j, i)] = iteration + base_tenure

            current_max = compute_max_dist(current)
            if current_max < best_max:
                best_max = current_max
                best = copy_routes(current)
                report_best_vrp(best)
                no_improve = 0

            # Diversification: if stuck, perturb by moving a few customers to different routes
            if no_improve > 30:
                # Perturb: remove a few from longest routes and reinsert
                dists = [route_dist(r) for r in current]
                max_dist = max(dists)
                critical = [i for i, d in enumerate(dists) if d >= 0.9 * max_dist]
                if not critical:
                    critical = [dists.index(max_dist)]
                to_remove = []
                for idx in critical:
                    route = current[idx]
                    if len(route) > 2:
                        # remove a random customer from route
                        pos = random.randint(1, len(route)-2)
                        to_remove.append(route[pos])
                        route.pop(pos)
                if to_remove:
                    # reinsert with regret-2? just simple insertion to minimize max
                    for cust in to_remove:
                        best_new_max = float('inf')
                        best_route_idx = -1
                        best_pos = -1
                        for idx, route in enumerate(current):
                            for ins_pos in range(1, len(route)):
                                new_route = route[:ins_pos] + [cust] + route[ins_pos:]
                                new_dist = route_dist(new_route)
                                other_max = max(route_dist(r) for j2, r in enumerate(current) if j2 != idx)
                                new_max = max(new_dist, other_max)
                                if new_max < best_new_max:
                                    best_new_max = new_max
                                    best_route_idx = idx
                                    best_pos = ins_pos
                        current[best_route_idx].insert(best_pos, cust)
                    current_max = compute_max_dist(current)
                    if current_max < best_max:
                        best_max = current_max
                        best = copy_routes(current)
                        report_best_vrp(best)
                    no_improve = 0

        return best

    random.seed(12345)
    initial = construct_initial()
    best = tabu_search(initial)
    while len(best) < truck_count:
        best.append([0, 0])
    report_best_vrp(best)
    return best