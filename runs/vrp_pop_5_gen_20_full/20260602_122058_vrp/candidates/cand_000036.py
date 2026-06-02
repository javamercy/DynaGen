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

    def compute_max_dist(routes):
        maxd = 0.0
        for r in routes:
            d = route_dist(r)
            if d > maxd:
                maxd = d
        return maxd

    def copy_routes(routes):
        return [r[:] for r in routes]

    # Construction from cand_000032
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

    # Regret-2 insertion (from cand_000032)
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

    # Perturbation: random removal and regret-2 reinsertion
    def perturb(routes):
        custom = copy_routes(routes)
        all_custs = list(range(1, n))
        random.shuffle(all_custs)
        remove_count = max(1, int(0.2 * (n-1)))
        to_remove = all_custs[:remove_count]
        removed = []
        for cust in to_remove:
            for route in custom:
                if cust in route:
                    route.remove(cust)
                    removed.append(cust)
                    break
        custom = regret_insert(custom, removed)
        return custom

    # Tabu search
    def tabu_search(initial_routes):
        current_routes = copy_routes(initial_routes)
        best_routes = copy_routes(initial_routes)
        best_max = compute_max_dist(best_routes)
        current_max = best_max

        max_iter = 100 + n * truck_count
        tabu_tenure = max(1, int(math.sqrt(n)))
        # tabu list: dict mapping move_key -> expiration_iter
        tabu = {}
        iteration = 0
        no_improve_iter = 0

        while iteration < max_iter:
            iteration += 1
            no_improve_iter += 1

            # Generate all candidate moves
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
                        # Best insertion position in route_j
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

                        # Check tabu
                        move_key = ('relocate', cust, i, j)
                        is_tabu = False
                        if move_key in tabu and tabu[move_key] > iteration:
                            is_tabu = True
                        # Aspiration: accept if new_max < best_max
                        if is_tabu and new_max >= best_max:
                            continue
                        if new_max < best_new_max or (new_max == best_new_max and (i, pos, j, best_ins_pos) < best_move_params if best_move_params else True):
                            best_new_max = new_max
                            best_move = ('relocate', i, pos, j, best_ins_pos)
                            best_move_type = 'relocate'
                            best_move_params = (i, pos, j, best_ins_pos)

            # Swap moves
            for i in range(truck_count):
                route_i = current_routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust_i = route_i[pos_i]
                    for j in range(i+1, truck_count):
                        route_j = current_routes[j]
                        if len(route_j) <= 2:
                            continue
                        for pos_j in range(1, len(route_j)-1):
                            cust_j = route_j[pos_j]
                            new_route_i = route_i[:]
                            new_route_i[pos_i] = cust_j
                            new_route_j = route_j[:]
                            new_route_j[pos_j] = cust_i
                            dist_i = route_dist(new_route_i)
                            dist_j = route_dist(new_route_j)
                            other_max = 0.0
                            for k, r in enumerate(current_routes):
                                if k == i or k == j:
                                    continue
                                other_max = max(other_max, route_dist(r))
                            new_max = max(other_max, dist_i, dist_j)

                            move_key = ('swap', cust_i, cust_j, i, j)
                            is_tabu = False
                            if move_key in tabu and tabu[move_key] > iteration:
                                is_tabu = True
                            if is_tabu and new_max >= best_max:
                                continue
                            if new_max < best_new_max or (new_max == best_new_max and (i, pos_i, j, pos_j) < best_move_params if best_move_params else True):
                                best_new_max = new_max
                                best_move = ('swap', i, pos_i, j, pos_j)
                                best_move_type = 'swap'
                                best_move_params = (i, pos_i, j, pos_j)

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
                        if new_max < best_new_max or (new_max == best_new_max and (i, a, b) < best_move_params if best_move_params else True):
                            best_new_max = new_max
                            best_move = ('2opt', i, a, b)
                            best_move_type = '2opt'
                            best_move_params = (i, a, b)

            # Cross-2-opt*
            for i in range(truck_count):
                route_i = current_routes[i]
                if len(route_i) <= 2:
                    continue
                for j in range(i+1, truck_count):
                    route_j = current_routes[j]
                    if len(route_j) <= 2:
                        continue
                    for cut_i in range(0, len(route_i)-1):
                        for cut_j in range(0, len(route_j)-1):
                            new_i = route_i[:cut_i+1] + route_j[cut_j+1:]
                            new_j = route_j[:cut_j+1] + route_i[cut_i+1:]
                            if len(new_i) < 2 or len(new_j) < 2:
                                continue
                            dist_i = route_dist(new_i)
                            dist_j = route_dist(new_j)
                            other_max = 0.0
                            for k, r in enumerate(current_routes):
                                if k == i or k == j:
                                    continue
                                other_max = max(other_max, route_dist(r))
                            new_max = max(other_max, dist_i, dist_j)

                            move_key = ('cross', i, j, cut_i, cut_j)
                            is_tabu = False
                            if move_key in tabu and tabu[move_key] > iteration:
                                is_tabu = True
                            if is_tabu and new_max >= best_max:
                                continue
                            if new_max < best_new_max or (new_max == best_new_max and (i, j, cut_i, cut_j) < best_move_params if best_move_params else True):
                                best_new_max = new_max
                                best_move = ('cross', i, j, cut_i, cut_j)
                                best_move_type = 'cross'
                                best_move_params = (i, j, cut_i, cut_j)

            if best_move is None:
                break

            # Apply best move
            if best_move_type == 'relocate':
                _, i, pos, j, ins_pos = best_move
                cust = current_routes[i][pos]
                current_routes[i] = current_routes[i][:pos] + current_routes[i][pos+1:]
                current_routes[j] = current_routes[j][:ins_pos] + [cust] + current_routes[j][ins_pos:]
                # Add tabu: reverse move
                tabu[('relocate', cust, j, i)] = iteration + tabu_tenure
            elif best_move_type == 'swap':
                _, i, pos_i, j, pos_j = best_move
                cust_i = current_routes[i][pos_i]
                cust_j = current_routes[j][pos_j]
                current_routes[i][pos_i] = cust_j
                current_routes[j][pos_j] = cust_i
                tabu[('swap', cust_i, cust_j, i, j)] = iteration + tabu_tenure
                tabu[('swap', cust_j, cust_i, j, i)] = iteration + tabu_tenure
            elif best_move_type == '2opt':
                _, i, a, b = best_move
                current_routes[i] = current_routes[i][:a] + current_routes[i][a:b+1][::-1] + current_routes[i][b+1:]
                tabu[('2opt', i, a, b)] = iteration + tabu_tenure
            elif best_move_type == 'cross':
                _, i, j, cut_i, cut_j = best_move
                orig_i = current_routes[i][:]
                orig_j = current_routes[j][:]
                current_routes[i] = orig_i[:cut_i+1] + orig_j[cut_j+1:]
                current_routes[j] = orig_j[:cut_j+1] + orig_i[cut_i+1:]
                tabu[('cross', i, j, cut_i, cut_j)] = iteration + tabu_tenure

            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = copy_routes(current_routes)
                report_best_vrp(best_routes)
                no_improve_iter = 0

            # Diversification if stuck
            if no_improve_iter >= 30:
                current_routes = perturb(best_routes)
                current_max = compute_max_dist(current_routes)
                tabu.clear()
                no_improve_iter = 0
                report_best_vrp(current_routes)

        return best_routes

    # Main
    random.seed(12345)
    initial = construct_initial()
    best = tabu_search(initial)
    while len(best) < truck_count:
        best.append([0, 0])
    report_best_vrp(best)
    return best