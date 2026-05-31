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

    def balance_routes(routes, lengths):
        improved = True
        while improved:
            improved = False
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_overall_reduction = 0
            best_pos = -1
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max_route)
                min_route = routes[min_idx]
                best_insertion_len = float('inf')
                best_ins_pos = -1
                for p in range(1, len(min_route)):
                    new_min_route = min_route[:p] + [cust] + min_route[p:]
                    l = route_distance(new_min_route)
                    if l < best_insertion_len:
                        best_insertion_len = l
                        best_ins_pos = p
                new_min_route = min_route[:best_ins_pos] + [cust] + min_route[best_ins_pos:]
                new_min_len = route_distance(new_min_route)
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_overall_reduction:
                    best_overall_reduction = reduction
                    best_cust = cust
                    best_pos = best_ins_pos
            if best_cust is not None:
                cust = best_cust
                new_max = [node for node in max_route if node != cust]
                min_route = routes[min_idx]
                new_min = min_route[:best_pos] + [cust] + min_route[best_pos:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_distance(new_max)
                lengths[min_idx] = route_distance(new_min)
                improved = True
        return routes, lengths

    best_routes = None
    best_max = float('inf')
    num_restarts = 3
    for restart in range(num_restarts):
        routes = regret_insertion_construction()
        lengths = [route_distance(r) for r in routes]
        routes, lengths = balance_routes(routes, lengths)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        max_iter = 200 * n  # total iterations per restart
        iteration = 0
        epsilon = 0.05  # initial threshold for acceptance
        while iteration < max_iter:
            # find current max route index
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            current_max = lengths[max_idx]
            best_move = None
            best_new_max = current_max
            best_total = sum(lengths)

            # Generate moves involving max_idx only
            # Inter-route relocate: move customer from max to other
            for pos in range(1, len(routes[max_idx])-1):
                cust = routes[max_idx][pos]
                new_src_route = routes[max_idx][:pos] + routes[max_idx][pos+1:]
                src_len = route_distance(new_src_route)
                for dst_idx in range(truck_count):
                    if dst_idx == max_idx:
                        continue
                    dst_route = routes[dst_idx]
                    for ins_pos in range(1, len(dst_route)):
                        new_dst_route = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                        new_lengths = lengths.copy()
                        new_lengths[max_idx] = src_len
                        new_lengths[dst_idx] = route_distance(new_dst_route)
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if (new_max < best_new_max or
                            (new_max == best_new_max and new_total < best_total) or
                            (new_max == best_new_max and new_total == best_total and max_idx < dst_idx)):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('relocate', max_idx, pos, dst_idx, ins_pos, new_src_route, new_dst_route)

            # Inter-route swap between max and another route
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                if len(other_route) <= 2:
                    continue
                for pos1 in range(1, len(routes[max_idx])-1):
                    cust_i = routes[max_idx][pos1]
                    for pos2 in range(1, len(other_route)-1):
                        cust_j = other_route[pos2]
                        new_i_route = routes[max_idx][:pos1] + [cust_j] + routes[max_idx][pos1+1:]
                        new_j_route = other_route[:pos2] + [cust_i] + other_route[pos2+1:]
                        new_lengths = lengths.copy()
                        new_lengths[max_idx] = route_distance(new_i_route)
                        new_lengths[other_idx] = route_distance(new_j_route)
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if (new_max < best_new_max or
                            (new_max == best_new_max and new_total < best_total) or
                            (new_max == best_new_max and new_total == best_total and max_idx < other_idx)):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('swap', max_idx, pos1, other_idx, pos2, new_i_route, new_j_route)

            # Intra-route 2-opt on max route
            if len(routes[max_idx]) > 3:
                for i in range(1, len(routes[max_idx])-2):
                    for j in range(i+1, len(routes[max_idx])-1):
                        new_route = routes[max_idx][:i] + routes[max_idx][i:j+1][::-1] + routes[max_idx][j+1:]
                        new_len = route_distance(new_route)
                        if new_len >= lengths[max_idx]:
                            continue
                        new_lengths = lengths.copy()
                        new_lengths[max_idx] = new_len
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if (new_max < best_new_max or
                            (new_max == best_new_max and new_total < best_total) or
                            (new_max == best_new_max and new_total == best_total and max_idx < 0)):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('2opt', max_idx, i, j, new_route)

            if best_move is not None and best_new_max < current_max:
                # apply move
                if best_move[0] == 'relocate':
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                elif best_move[0] == 'swap':
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                elif best_move[0] == '2opt':
                    routes[best_move[1]] = best_move[4]
                lengths = [route_distance(r) for r in routes]
                current_max = max(lengths)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                iteration += 1
            else:
                # Shake: remove up to 2 customers from max route and reinsert via regret-2
                max_idx = max(range(truck_count), key=lambda i: lengths[i])
                max_route = routes[max_idx]
                if len(max_route) <= 3:
                    # nothing to remove, break or perturb elsewhere
                    # just random perturbation
                    pass
                else:
                    # remove one or two customers
                    remove_count = min(2, len(max_route)-2)
                    remove_positions = random.sample(range(1, len(max_route)-1), remove_count)
                    remove_positions.sort(reverse=True)
                    removed = []
                    for pos in remove_positions:
                        removed.append(max_route.pop(pos))
                    # now reinsert removed customers using regret-2 on all routes
                    # temporarily mark max_route as changed
                    unvisited = removed
                    while unvisited:
                        best_cust = None
                        best_regret = -float('inf')
                        best_inc = float('inf')
                        best_route_idx = -1
                        best_pos = -1
                        for cust in unvisited:
                            incs = []
                            for r_idx, route in enumerate(routes):
                                # consider all routes including max_idx
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
                    # after reinsertion, recompute lengths and balance
                    lengths = [route_distance(r) for r in routes]
                    routes, lengths = balance_routes(routes, lengths)
                    current_max = max(lengths)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                # also try a threshold acceptance: if no improvement after shake, maybe accept a slight degradation
                if best_move is None or best_new_max >= current_max:
                    # accept a worse solution with probability decreasing with epsilon
                    if current_max <= best_max * (1 + epsilon):
                        # keep current solution (already applied shake)
                        pass
                    else:
                        # if too bad, maybe revert? We'll just continue with current
                        pass
                    epsilon *= 0.99  # decay
                iteration += 1
    return best_routes