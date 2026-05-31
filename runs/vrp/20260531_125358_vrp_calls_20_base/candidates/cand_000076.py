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
                new_min_route = min_route[:best_pos] + [cust] + min_route[best_pos:]
                new_min_len = route_distance(new_min_route)
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
                if regret > best_regret or (regret == best_regret and (inc < best_inc or (inc == best_inc and cust < best_cust))):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes

    best_routes = None
    best_max = float('inf')
    num_restarts = max(1, min(5, n // 10))
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
        for iteration in range(max_iter):
            # Variable Neighborhood Descent (sequentially, restart on improvement)
            improved = True
            while improved:
                improved = False
                # Neighborhood 1: Inter-route relocate
                for cust in range(1, n):
                    src_idx = None
                    src_pos = None
                    for r_idx, route in enumerate(routes):
                        if cust in route:
                            src_idx = r_idx
                            src_pos = route.index(cust)
                            break
                    if src_idx is None:
                        continue
                    new_src_route = routes[src_idx][:src_pos] + routes[src_idx][src_pos+1:]
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
                            if new_max < current_max:
                                current_max = new_max
                                routes[src_idx] = new_src_route
                                routes[dst_idx] = new_dst_route
                                lengths = new_lengths
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue

                # Neighborhood 2: Inter-route swap
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
                                if new_max < current_max:
                                    current_max = new_max
                                    routes[i_idx] = new_i_route
                                    routes[j_idx] = new_j_route
                                    lengths = new_lengths
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue

                # Neighborhood 3: Intra-route 2-opt
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_len = route_distance(new_route)
                            if new_len < lengths[r_idx]:
                                new_lengths = lengths[:]
                                new_lengths[r_idx] = new_len
                                new_max = max(new_lengths)
                                if new_max < current_max:
                                    current_max = new_max
                                    routes[r_idx] = new_route
                                    lengths = new_lengths
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)

            # Deterministic perturbation: remove customer with highest reduction from longest route and reinsert cheapest
            worst_route_idx = max(range(truck_count), key=lambda i: lengths[i])
            worst_route = routes[worst_route_idx]
            best_saving = 0
            best_cust = None
            best_pos_removed = None
            for pos in range(1, len(worst_route)-1):
                cust = worst_route[pos]
                saving = distance_matrix[worst_route[pos-1], cust] + distance_matrix[cust, worst_route[pos+1]] - distance_matrix[worst_route[pos-1], worst_route[pos+1]]
                if saving > best_saving:
                    best_saving = saving
                    best_cust = cust
                    best_pos_removed = pos
            if best_cust is not None:
                new_worst_route = worst_route[:best_pos_removed] + worst_route[best_pos_removed+1:]
                routes[worst_route_idx] = new_worst_route
                # Reinsert with cheapest insertion
                best_total_inc = float('inf')
                best_route_idx = -1
                best_pos = -1
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], best_cust] + distance_matrix[best_cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        if inc < best_total_inc:
                            best_total_inc = inc
                            best_route_idx = r_idx
                            best_pos = pos
                if best_route_idx >= 0:
                    routes[best_route_idx].insert(best_pos, best_cust)
                # Apply 2-opt on affected routes
                for r_idx in [worst_route_idx, best_route_idx]:
                    if len(routes[r_idx]) > 2:
                        routes[r_idx] = two_opt(routes[r_idx], max_iter=5)
                lengths = [route_distance(r) for r in routes]
                current_max = max(lengths)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)

    return best_routes