import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def regret_insertion_construction():
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
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes

    def initial_balance(routes):
        lengths = [route_distance(r) for r in routes]
        max_idx = max(range(truck_count), key=lambda i: lengths[i])
        min_idx = min(range(truck_count), key=lambda i: lengths[i])
        if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
            return routes, lengths
        max_route = routes[max_idx]
        best_cust = None
        best_pos = -1
        best_reduction = 0
        for pos in range(1, len(max_route)-1):
            cust = max_route[pos]
            new_max_len = route_distance(max_route[:pos] + max_route[pos+1:])
            min_route = routes[min_idx]
            for ins_pos in range(1, len(min_route)):
                new_min_len = route_distance(min_route[:ins_pos] + [cust] + min_route[ins_pos:])
                old_max = max(lengths)
                new_max = max(new_max_len, new_min_len, *[lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)])
                reduction = old_max - new_max
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_cust = cust
                    best_pos = ins_pos
        if best_cust is not None:
            new_max_route = [x for x in max_route if x != best_cust]
            new_min_route = routes[min_idx][:best_pos] + [best_cust] + routes[min_idx][best_pos:]
            routes[max_idx] = new_max_route
            routes[min_idx] = new_min_route
            lengths = [route_distance(r) for r in routes]
        return routes, lengths

    def get_longest_indices(lengths):
        max_len = max(lengths)
        return [i for i, l in enumerate(lengths) if l == max_len]

    def steepest_descent_max(routes, lengths):
        improved = True
        while improved:
            improved = False
            best_move = None
            best_new_max = max(lengths)
            best_total = sum(lengths)
            longest_indices = get_longest_indices(lengths)

            # Inter-route relocate from a longest route to any other
            for src_idx in longest_indices:
                src_route = routes[src_idx]
                if len(src_route) <= 2:
                    continue
                for pos in range(1, len(src_route)-1):
                    cust = src_route[pos]
                    new_src = src_route[:pos] + src_route[pos+1:]
                    src_len = route_distance(new_src)
                    for dst_idx in range(truck_count):
                        if dst_idx == src_idx:
                            continue
                        dst_route = routes[dst_idx]
                        for ins_pos in range(1, len(dst_route)):
                            new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                            dst_len = route_distance(new_dst)
                            new_lengths = lengths[:]
                            new_lengths[src_idx] = src_len
                            new_lengths[dst_idx] = dst_len
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if (new_max < best_new_max or
                                (new_max == best_new_max and new_total < best_total) or
                                (new_max == best_new_max and new_total == best_total and src_idx < dst_idx)):
                                best_new_max = new_max
                                best_total = new_total
                                best_move = ('relocate', src_idx, pos, dst_idx, ins_pos, new_src, new_dst)

            # Inter-route swap between a longest route and another
            for i_idx in longest_indices:
                i_route = routes[i_idx]
                if len(i_route) <= 2:
                    continue
                for i_pos in range(1, len(i_route)-1):
                    cust_i = i_route[i_pos]
                    for j_idx in range(truck_count):
                        if j_idx == i_idx:
                            continue
                        j_route = routes[j_idx]
                        if len(j_route) <= 2:
                            continue
                        for j_pos in range(1, len(j_route)-1):
                            cust_j = j_route[j_pos]
                            new_i = i_route[:i_pos] + [cust_j] + i_route[i_pos+1:]
                            new_j = j_route[:j_pos] + [cust_i] + j_route[j_pos+1:]
                            i_len = route_distance(new_i)
                            j_len = route_distance(new_j)
                            new_lengths = lengths[:]
                            new_lengths[i_idx] = i_len
                            new_lengths[j_idx] = j_len
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if (new_max < best_new_max or
                                (new_max == best_new_max and new_total < best_total) or
                                (new_max == best_new_max and new_total == best_total and i_idx < j_idx)):
                                best_new_max = new_max
                                best_total = new_total
                                best_move = ('swap', i_idx, i_pos, j_idx, j_pos, new_i, new_j)

            # Intra-route 2-opt on longest routes
            for r_idx in longest_indices:
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
                        if (new_max < best_new_max or
                            (new_max == best_new_max and new_total < best_total)):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('2opt', r_idx, i, j, new_route)

            if best_move is not None and best_new_max < max(lengths):
                if best_move[0] == 'relocate':
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                elif best_move[0] == 'swap':
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                elif best_move[0] == '2opt':
                    routes[best_move[1]] = best_move[4]
                lengths = [route_distance(r) for r in routes]
                improved = True
        return routes, lengths

    def guided_perturbation(routes, lengths):
        max_idx = max(range(truck_count), key=lambda i: lengths[i])
        max_route = routes[max_idx]
        n_remove = min(3, len(max_route)-2)
        if n_remove == 0:
            return routes, lengths
        remove_positions = random.sample(range(1, len(max_route)-1), n_remove)
        removed_customers = [max_route[pos] for pos in reversed(sorted(remove_positions))]
        new_max_route = [node for i, node in enumerate(max_route) if i not in remove_positions]
        routes[max_idx] = new_max_route
        lengths[max_idx] = route_distance(new_max_route)

        # Reinsert removed customers using regret-2 into other routes
        unvisited = set(removed_customers)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(routes):
                    if r_idx == max_idx and len(route) <= 2:
                        continue
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        incs.append((inc, pos, r_idx))
                if not incs:
                    continue
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
            if best_cust is None:
                break
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
            lengths[best_route_idx] = route_distance(routes[best_route_idx])
        return routes, lengths

    best_routes = None
    best_max = float('inf')
    num_restarts = max(1, min(3, n // 10))
    for restart in range(num_restarts):
        routes = regret_insertion_construction()
        lengths = [route_distance(r) for r in routes]
        routes, lengths = initial_balance(routes, lengths)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        max_iter = n * truck_count * 2
        for iteration in range(max_iter):
            routes, lengths = steepest_descent_max(routes, lengths)
            new_max = max(lengths)
            if new_max < current_max:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
            else:
                # Perturbation
                routes, lengths = guided_perturbation(routes, lengths)
                # Apply 2-opt to all routes
                for idx in range(truck_count):
                    if len(routes[idx]) > 2:
                        improved = True
                        while improved:
                            improved = False
                            route = routes[idx]
                            for i in range(1, len(route)-2):
                                for j in range(i+1, len(route)-1):
                                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                    if route_distance(new_route) < route_distance(route):
                                        route = new_route
                                        improved = True
                            routes[idx] = route
                        lengths[idx] = route_distance(routes[idx])
                current_max = max(lengths)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)

    return best_routes