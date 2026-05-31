import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def two_opt(route, max_iter=10):
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

    def balance_routes(routes, lengths, max_bal_iters=20):
        improved = True
        bal_it = 0
        while improved and bal_it < max_bal_iters:
            improved = False
            bal_it += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_overall_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max_route)
                min_route = routes[min_idx]
                best_insertion_len = float('inf')
                best_pos = -1
                for p in range(1, len(min_route)):
                    new_min_route = min_route[:p] + [cust] + min_route[p:]
                    l = route_distance(new_min_route)
                    if l < best_insertion_len:
                        best_insertion_len = l
                        best_pos = p
                new_min_len = best_insertion_len
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_overall_reduction:
                    best_overall_reduction = reduction
                    best_cust = (cust, best_pos)
            if best_cust is not None:
                cust, best_insert_pos = best_cust
                new_max = [node for node in max_route if node != cust]
                min_route = routes[min_idx]
                new_min = min_route[:best_insert_pos] + [cust] + min_route[best_insert_pos:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_distance(new_max)
                lengths[min_idx] = route_distance(new_min)
                improved = True
        return routes, lengths

    def regret_insertion_construction(k=3):
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

    def ruin_recreate(routes, lengths, fraction=0.2):
        n_cust = n - 1
        num_remove = max(1, int(n_cust * fraction))
        custs = list(range(1, n))
        random.shuffle(custs)
        to_remove = custs[:num_remove]
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
                if len(incs) >= 3:
                    regret = incs[1][0] - incs[0][0] + incs[2][0] - incs[0][0]
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
    num_restarts = max(1, min(3, n // 10))
    for restart in range(num_restarts):
        routes = regret_insertion_construction(k=3)
        lengths = [route_distance(r) for r in routes]
        routes, lengths = balance_routes(routes, lengths)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        max_iter = n * truck_count * 2
        stagnation_limit = max(5, int(0.05 * max_iter))
        stagnation = 0
        for iteration in range(max_iter):
            # VND in fixed order: relocate, swap, 2-opt
            improved = False
            # Relocate
            best_move = None
            best_new_max = current_max
            best_total = sum(lengths)
            for src_idx in range(truck_count):
                src_route = routes[src_idx]
                if len(src_route) <= 2:
                    continue
                for pos in range(1, len(src_route)-1):
                    cust = src_route[pos]
                    new_src_route = src_route[:pos] + src_route[pos+1:]
                    src_len = route_distance(new_src_route)
                    for dst_idx in range(truck_count):
                        if dst_idx == src_idx:
                            continue
                        dst_route = routes[dst_idx]
                        if len(dst_route) <= 2:
                            continue
                        for ins_pos in range(1, len(dst_route)):
                            new_dst_route = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                            new_lengths = lengths[:]
                            new_lengths[src_idx] = src_len
                            new_lengths[dst_idx] = route_distance(new_dst_route)
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if (new_max < best_new_max or
                                (new_max == best_new_max and new_total < best_total) or
                                (new_max == best_new_max and new_total == best_total and (src_idx, pos, dst_idx, ins_pos) < (best_move[1], best_move[2], best_move[3], best_move[4]) if best_move else True)):
                                best_new_max = new_max
                                best_total = new_total
                                best_move = ('relocate', src_idx, pos, dst_idx, ins_pos, new_src_route, new_dst_route)
            if best_move and best_new_max < current_max:
                improved = True
                routes[best_move[1]] = best_move[5]
                routes[best_move[3]] = best_move[6]
                lengths = [route_distance(r) for r in routes]
                current_max = max(lengths)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
            # Swap (if no improvement from relocate)
            if not improved:
                best_move = None
                best_new_max = current_max
                best_total = sum(lengths)
                for i_idx in range(truck_count):
                    i_route = routes[i_idx]
                    if len(i_route) <= 2:
                        continue
                    for i_pos in range(1, len(i_route)-1):
                        cust_i = i_route[i_pos]
                        for j_idx in range(i_idx+1, truck_count):
                            j_route = routes[j_idx]
                            if len(j_route) <= 2:
                                continue
                            for j_pos in range(1, len(j_route)-1):
                                cust_j = j_route[j_pos]
                                new_i_route = i_route[:i_pos] + [cust_j] + i_route[i_pos+1:]
                                new_j_route = j_route[:j_pos] + [cust_i] + j_route[j_pos+1:]
                                new_lengths = lengths[:]
                                new_lengths[i_idx] = route_distance(new_i_route)
                                new_lengths[j_idx] = route_distance(new_j_route)
                                new_max = max(new_lengths)
                                new_total = sum(new_lengths)
                                if (new_max < best_new_max or
                                    (new_max == best_new_max and new_total < best_total) or
                                    (new_max == best_new_max and new_total == best_total and (i_idx, i_pos, j_idx, j_pos) < (best_move[1], best_move[2], best_move[3], best_move[4]) if best_move else True)):
                                    best_new_max = new_max
                                    best_total = new_total
                                    best_move = ('swap', i_idx, i_pos, j_idx, j_pos, new_i_route, new_j_route)
                if best_move and best_new_max < current_max:
                    improved = True
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                    lengths = [route_distance(r) for r in routes]
                    current_max = max(lengths)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
            # 2-opt (if no improvement from swap)
            if not improved:
                best_move = None
                best_new_max = current_max
                best_total = sum(lengths)
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
                            if (new_max < best_new_max or
                                (new_max == best_new_max and new_total < best_total) or
                                (new_max == best_new_max and new_total == best_total and (r_idx, i, j) < (best_move[1], best_move[2], best_move[3]) if best_move else True)):
                                best_new_max = new_max
                                best_total = new_total
                                best_move = ('2opt', r_idx, i, j, new_route)
                if best_move and best_new_max < current_max:
                    improved = True
                    routes[best_move[1]] = best_move[4]
                    lengths = [route_distance(r) for r in routes]
                    current_max = max(lengths)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
            if improved:
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= stagnation_limit:
                    routes, lengths = ruin_recreate(routes, lengths, fraction=0.2)
                    current_max = max(lengths)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                    stagnation = 0
    return best_routes