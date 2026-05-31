import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

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
            else:
                break
        return routes, lengths

    def regret_insertion_construction(rng):
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            candidates = []
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
                candidates.append((regret, inc, cust, r_idx, pos))
            candidates.sort(key=lambda x: (-x[0], x[1]))
            rcl_size = min(2, len(candidates))
            rcl = candidates[:rcl_size]
            if len(rcl) > 1:
                chosen = rng.choice(rcl)
            else:
                chosen = rcl[0]
            _, _, cust, r_idx, pos = chosen
            routes[r_idx].insert(pos, cust)
            unvisited.remove(cust)
        return routes

    best_routes = None
    best_max = float('inf')
    num_restarts = 5
    for restart in range(num_restarts):
        rng = random.Random(restart)
        routes = regret_insertion_construction(rng)
        lengths = [route_distance(r) for r in routes]
        routes, lengths = balance_routes(routes, lengths)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        max_iter = n * truck_count * 2
        for iteration in range(max_iter):
            improved = False
            # Inter-route relocate - first improvement
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
                        new_total = sum(new_lengths)
                        if new_max < current_max or (new_max == current_max and new_total < current_total):
                            routes[src_idx] = new_src_route
                            routes[dst_idx] = new_dst_route
                            lengths = new_lengths
                            current_max = new_max
                            current_total = new_total
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                # Inter-route swap - first improvement
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
                                if new_max < current_max or (new_max == current_max and new_total < current_total):
                                    routes[i_idx] = new_i_route
                                    routes[j_idx] = new_j_route
                                    lengths = new_lengths
                                    current_max = new_max
                                    current_total = new_total
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                # Intra-route 2-opt - first improvement
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
                            if new_max < current_max or (new_max == current_max and new_total < current_total):
                                routes[r_idx] = new_route
                                lengths = new_lengths
                                current_max = new_max
                                current_total = new_total
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
            else:
                # Perturbation: ruin-and-recreate
                all_customers = list(range(1, n))
                num_remove = max(1, int(0.2 * len(all_customers)))
                to_remove = rng.sample(all_customers, k=num_remove)
                for cust in to_remove:
                    for r_idx, route in enumerate(routes):
                        if cust in route:
                            route.remove(cust)
                            break
                unvisited = set(to_remove)
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
                lengths = [route_distance(r) for r in routes]
                routes, lengths = balance_routes(routes, lengths)
                current_max = max(lengths)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
    return best_routes