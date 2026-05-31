import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def two_opt(route, max_iter=5):
        route = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
        return route

    def balance_routes(routes, lengths):
        improved = True
        it = 0
        while improved and it < n:
            improved = False
            it += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_reduction = 0
            best_insert_pos = -1
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max_route)
                min_route = routes[min_idx]
                best_pos = -1
                best_min_len = float('inf')
                for p in range(1, len(min_route)):
                    new_min_route = min_route[:p] + [cust] + min_route[p:]
                    l = route_distance(new_min_route)
                    if l < best_min_len:
                        best_min_len = l
                        best_pos = p
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, best_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_cust = cust
                    best_insert_pos = best_pos
            if best_cust is not None and best_reduction > 0:
                routes[max_idx] = [node for node in max_route if node != best_cust]
                min_route = routes[min_idx]
                routes[min_idx] = min_route[:best_insert_pos] + [best_cust] + min_route[best_insert_pos:]
                lengths[max_idx] = route_distance(routes[max_idx])
                lengths[min_idx] = route_distance(routes[min_idx])
                improved = True
        return routes, lengths

    def regret_insertion_construction(k=2):
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        incs.append((inc, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                if len(incs) >= k:
                    regret = sum(incs[i][0] - incs[0][0] for i in range(1, k))
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret or (regret == best_regret and inc < best_inc):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes

    def ruin_recreate_random(routes, lengths, fraction=0.15):
        n_cust = n - 1
        num_remove = max(1, int(n_cust * fraction))
        all_customers = [c for route in routes for c in route[1:-1]]
        random.shuffle(all_customers)
        to_remove = all_customers[:num_remove]
        new_routes = [[0, 0] for _ in range(truck_count)]
        for r_idx, route in enumerate(routes):
            new_routes[r_idx] = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
        unvisited = set(to_remove)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(new_routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        incs.append((inc, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                if len(incs) >= 2:
                    regret = incs[1][0] - incs[0][0]
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret or (regret == best_regret and inc < best_inc):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            new_routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        for r_idx in range(truck_count):
            if len(new_routes[r_idx]) > 2:
                new_routes[r_idx] = two_opt(new_routes[r_idx], max_iter=5)
        new_lengths = [route_distance(r) for r in new_routes]
        new_routes, new_lengths = balance_routes(new_routes, new_lengths)
        return new_routes, new_lengths

    best_routes = None
    best_max = float('inf')
    best_total = float('inf')
    routes = regret_insertion_construction(k=2)
    lengths = [route_distance(r) for r in routes]
    routes, lengths = balance_routes(routes, lengths)
    current_max = max(lengths)
    current_total = sum(lengths)
    if (current_max < best_max) or (current_max == best_max and current_total < best_total):
        best_max = current_max
        best_total = current_total
        best_routes = [r[:] for r in routes]
        report_best_vrp(best_routes)

    max_iter = n * truck_count * 2
    stagnation_counter = 0
    ruin_fraction = 0.15
    for iteration in range(max_iter):
        best_move = None
        best_new_max = current_max
        best_new_total = current_total
        best_tie_break = (current_max, current_total, 0)

        # Inter-route relocate
        for cust in range(1, n):
            src_idx = None
            for r_idx, route in enumerate(routes):
                if cust in route:
                    src_idx = r_idx
                    break
            if src_idx is None:
                continue
            src_pos = routes[src_idx].index(cust)
            new_src_route = routes[src_idx][:src_pos] + routes[src_idx][src_pos+1:]
            src_len = route_distance(new_src_route)
            for dst_idx in range(truck_count):
                if dst_idx == src_idx:
                    continue
                dst_route = routes[dst_idx]
                for ins_pos in range(1, len(dst_route)):
                    new_dst_route = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                    new_lengths = lengths[:]
                    new_lengths[src_idx] = src_len
                    new_lengths[dst_idx] = route_distance(new_dst_route)
                    new_max = max(new_lengths)
                    new_total = sum(new_lengths)
                    tie_break = (new_max, new_total, src_idx, dst_idx, ins_pos)
                    if (new_max < best_new_max) or (new_max == best_new_max and new_total < best_new_total) or (new_max == best_new_max and new_total == best_new_total and tie_break[2:] < best_tie_break[2:]):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_tie_break = tie_break
                        best_move = ('relocate', cust, src_idx, src_pos, dst_idx, ins_pos, new_src_route, new_dst_route)

        # Inter-route swap
        for i_idx in range(truck_count):
            i_route = routes[i_idx]
            for i_pos in range(1, len(i_route)-1):
                cust_i = i_route[i_pos]
                for j_idx in range(i_idx+1, truck_count):
                    j_route = routes[j_idx]
                    for j_pos in range(1, len(j_route)-1):
                        cust_j = j_route[j_pos]
                        new_i_route = i_route[:i_pos] + [cust_j] + i_route[i_pos+1:]
                        new_j_route = j_route[:j_pos] + [cust_i] + j_route[j_pos+1:]
                        new_lengths = lengths[:]
                        new_lengths[i_idx] = route_distance(new_i_route)
                        new_lengths[j_idx] = route_distance(new_j_route)
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        tie_break = (new_max, new_total, i_idx, j_idx, i_pos, j_pos)
                        if (new_max < best_new_max) or (new_max == best_new_max and new_total < best_new_total) or (new_max == best_new_max and new_total == best_new_total and tie_break[2:] < best_tie_break[2:]):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_tie_break = tie_break
                            best_move = ('swap', i_idx, i_pos, j_idx, j_pos, new_i_route, new_j_route)

        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_len = route_distance(new_route)
                    if new_len >= lengths[r_idx]:
                        continue
                    new_lengths = lengths[:]
                    new_lengths[r_idx] = new_len
                    new_max = max(new_lengths)
                    new_total = sum(new_lengths)
                    tie_break = (new_max, new_total, r_idx, i, j)
                    if (new_max < best_new_max) or (new_max == best_new_max and new_total < best_new_total) or (new_max == best_new_max and new_total == best_new_total and tie_break[2:] < best_tie_break[2:]):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_tie_break = tie_break
                        best_move = ('2opt', r_idx, i, j, new_route)

        if best_move is not None and (best_new_max < current_max or (best_new_max == current_max and best_new_total < current_total)):
            if best_move[0] == 'relocate':
                routes[best_move[2]] = best_move[6]
                routes[best_move[4]] = best_move[7]
            elif best_move[0] == 'swap':
                routes[best_move[1]] = best_move[5]
                routes[best_move[3]] = best_move[6]
            elif best_move[0] == '2opt':
                routes[best_move[1]] = best_move[4]
            lengths = [route_distance(r) for r in routes]
            current_max = max(lengths)
            current_total = sum(lengths)
            if (current_max < best_max) or (current_max == best_max and current_total < best_total):
                best_max = current_max
                best_total = current_total
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
            stagnation_counter = 0
            ruin_fraction = 0.15
        else:
            stagnation_counter += 1
            ruin_fraction = min(0.3, 0.15 + 0.05 * stagnation_counter)
            routes, lengths = ruin_recreate_random(routes, lengths, fraction=ruin_fraction)
            current_max = max(lengths)
            current_total = sum(lengths)
            if (current_max < best_max) or (current_max == best_max and current_total < best_total):
                best_max = current_max
                best_total = current_total
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
            for r_idx in range(truck_count):
                if len(routes[r_idx]) > 2:
                    routes[r_idx] = two_opt(routes[r_idx], max_iter=5)
            lengths = [route_distance(r) for r in routes]
            current_max = max(lengths)
            current_total = sum(lengths)
            if (current_max < best_max) or (current_max == best_max and current_total < best_total):
                best_max = current_max
                best_total = current_total
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
    return best_routes