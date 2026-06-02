import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    max_iter = max(5, int(5 * n / truck_count + 10))
    best_max = float('inf')
    best_routes = None
    
    def compute_route_length(route):
        if len(route) < 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def initial_solution():
        shuffled = customers[:]
        random.shuffle(shuffled)
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        unassigned = set(shuffled)
        while unassigned:
            best_cust = None
            best_route = -1
            best_pos = -1
            best_regret = -1.0
            best_new_max = float('inf')
            best_route_len = float('inf')
            for c in unassigned:
                insertions = []
                for ridx, route in enumerate(routes):
                    best_inc = float('inf')
                    best_pos_local = -1
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        inc = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        new_len = lengths[ridx] + inc
                        new_max = max(lengths[:ridx] + [new_len] + lengths[ridx+1:])
                        if new_max < best_inc or (new_max == best_inc and lengths[ridx] < best_route_len):
                            best_inc = new_max
                            best_pos_local = pos
                    insertions.append((best_inc, best_pos_local, ridx))
                insertions.sort(key=lambda x: (x[0], lengths[x[2]], x[1]))
                if not insertions:
                    continue
                best_cost = insertions[0][0]
                second_cost = insertions[1][0] if len(insertions) > 1 else best_cost
                regret = second_cost - best_cost
                if regret > best_regret or (regret == best_regret and (best_cost < best_new_max or (best_cost == best_new_max and lengths[insertions[0][2]] < best_route_len))):
                    best_regret = regret
                    best_cust = c
                    best_new_max = best_cost
                    best_route = insertions[0][2]
                    best_pos = insertions[0][1]
                    best_route_len = lengths[best_route]
            if best_cust is None:
                best_cust = next(iter(unassigned))
                best_new_max = float('inf')
                best_route = -1
                best_pos = -1
                best_route_len = float('inf')
                for ridx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        inc = distance_matrix[prev, best_cust] + distance_matrix[best_cust, nxt] - distance_matrix[prev, nxt]
                        new_len = lengths[ridx] + inc
                        new_max = max(lengths[:ridx] + [new_len] + lengths[ridx+1:])
                        if new_max < best_new_max or (new_max == best_new_max and lengths[ridx] < best_route_len):
                            best_new_max = new_max
                            best_route = ridx
                            best_pos = pos
                            best_route_len = lengths[ridx]
            route = routes[best_route]
            route.insert(best_pos, best_cust)
            lengths[best_route] = compute_route_length(route)
            unassigned.remove(best_cust)
        return routes, lengths
    
    def local_search(routes, lengths):
        improved = True
        max_iter_ls = 5 * n * truck_count + 10
        while improved and max_iter_ls > 0:
            improved = False
            max_iter_ls -= 1
            # 2-opt
            for ridx in range(truck_count):
                route = routes[ridx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-9:
                            old_len = lengths[ridx]
                            new_len = old_len - (old - new)
                            old_max = max(lengths)
                            new_max = max(lengths[:ridx] + [new_len] + lengths[ridx+1:])
                            if new_max < old_max - 1e-9:
                                route[i:j+1] = reversed(route[i:j+1])
                                lengths[ridx] = new_len
                                improved = True
                                report_best_vrp(routes)
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # relocate
            for rfrom in range(truck_count):
                route_from = routes[rfrom]
                if len(route_from) <= 2:
                    continue
                for idxc in range(1, len(route_from)-1):
                    c = route_from[idxc]
                    prev = route_from[idxc-1]
                    nxt = route_from[idxc+1]
                    cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                    new_len_from = lengths[rfrom] - cost_remove
                    for rto in range(truck_count):
                        if rto == rfrom:
                            continue
                        route_to = routes[rto]
                        for pos in range(1, len(route_to)):
                            prev_to = route_to[pos-1]
                            nxt_to = route_to[pos]
                            cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                            new_len_to = lengths[rto] + cost_insert
                            old_max = max(lengths)
                            new_max = max(lengths[:rfrom] + [new_len_from] + lengths[rfrom+1:rto] + [new_len_to] + lengths[rto+1:])
                            if new_max < old_max - 1e-9:
                                route_from.pop(idxc)
                                lengths[rfrom] = new_len_from
                                route_to.insert(pos, c)
                                lengths[rto] = new_len_to
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
            # swap
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for idx1 in range(1, len(route1)-1):
                    c1 = route1[idx1]
                    prev1 = route1[idx1-1]
                    nxt1 = route1[idx1+1]
                    cost_remove1 = distance_matrix[prev1, c1] + distance_matrix[c1, nxt1] - distance_matrix[prev1, nxt1]
                    for r2 in range(r1+1, truck_count):
                        route2 = routes[r2]
                        if len(route2) <= 2:
                            continue
                        for idx2 in range(1, len(route2)-1):
                            c2 = route2[idx2]
                            prev2 = route2[idx2-1]
                            nxt2 = route2[idx2+1]
                            cost_remove2 = distance_matrix[prev2, c2] + distance_matrix[c2, nxt2] - distance_matrix[prev2, nxt2]
                            cost_insert1 = distance_matrix[prev1, c2] + distance_matrix[c2, nxt1] - distance_matrix[prev1, nxt1]
                            new_len1 = lengths[r1] - cost_remove1 + cost_insert1
                            cost_insert2 = distance_matrix[prev2, c1] + distance_matrix[c1, nxt2] - distance_matrix[prev2, nxt2]
                            new_len2 = lengths[r2] - cost_remove2 + cost_insert2
                            old_max = max(lengths)
                            new_max = max(lengths[:r1] + [new_len1] + lengths[r1+1:r2] + [new_len2] + lengths[r2+1:])
                            if new_max < old_max - 1e-9:
                                del route1[idx1]
                                del route2[idx2]
                                route1.insert(idx1, c2)
                                route2.insert(idx2, c1)
                                lengths[r1] = new_len1
                                lengths[r2] = new_len2
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes, lengths
    
    # VNS with restart
    routes, lengths = initial_solution()
    routes, lengths = local_search(routes, lengths)
    cur_max = max(lengths)
    if cur_max < best_max:
        best_max = cur_max
        best_routes = [route[:] for route in routes]
    no_improve = 0
    for it in range(max_iter):
        removal_frac = 0.1 + 0.2 * (it / max_iter)
        num_remove = max(1, int(removal_frac * (n - 1)))
        # shake: remove random customers
        all_cust = [c for route in routes for c in route[1:-1]]
        if len(all_cust) <= num_remove:
            continue
        removed = random.sample(all_cust, num_remove)
        for c in removed:
            for ridx, route in enumerate(routes):
                if c in route:
                    idx = route.index(c)
                    prev = route[idx-1]
                    nxt = route[idx+1]
                    cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                    route.pop(idx)
                    lengths[ridx] -= cost_remove
                    break
        # repair
        unassigned = set(removed)
        while unassigned:
            best_cust = None
            best_route = -1
            best_pos = -1
            best_regret = -1.0
            best_new_max = float('inf')
            best_route_len = float('inf')
            for c in unassigned:
                insertions = []
                for ridx, route in enumerate(routes):
                    best_inc = float('inf')
                    best_pos_local = -1
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        inc = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        new_len = lengths[ridx] + inc
                        new_max = max(lengths[:ridx] + [new_len] + lengths[ridx+1:])
                        if new_max < best_inc or (new_max == best_inc and lengths[ridx] < best_route_len):
                            best_inc = new_max
                            best_pos_local = pos
                    insertions.append((best_inc, best_pos_local, ridx))
                insertions.sort(key=lambda x: (x[0], lengths[x[2]], x[1]))
                if not insertions:
                    continue
                best_cost = insertions[0][0]
                second_cost = insertions[1][0] if len(insertions) > 1 else best_cost
                regret = second_cost - best_cost
                if regret > best_regret or (regret == best_regret and (best_cost < best_new_max or (best_cost == best_new_max and lengths[insertions[0][2]] < best_route_len))):
                    best_regret = regret
                    best_cust = c
                    best_new_max = best_cost
                    best_route = insertions[0][2]
                    best_pos = insertions[0][1]
                    best_route_len = lengths[best_route]
            if best_cust is None:
                best_cust = next(iter(unassigned))
                best_new_max = float('inf')
                best_route = -1
                best_pos = -1
                best_route_len = float('inf')
                for ridx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        inc = distance_matrix[prev, best_cust] + distance_matrix[best_cust, nxt] - distance_matrix[prev, nxt]
                        new_len = lengths[ridx] + inc
                        new_max = max(lengths[:ridx] + [new_len] + lengths[ridx+1:])
                        if new_max < best_new_max or (new_max == best_new_max and lengths[ridx] < best_route_len):
                            best_new_max = new_max
                            best_route = ridx
                            best_pos = pos
                            best_route_len = lengths[ridx]
            route = routes[best_route]
            route.insert(best_pos, best_cust)
            lengths[best_route] = compute_route_length(route)
            unassigned.remove(best_cust)
        # local search
        routes, lengths = local_search(routes, lengths)
        new_max = max(lengths)
        if new_max < cur_max - 1e-9:
            cur_max = new_max
            no_improve = 0
            if new_max < best_max - 1e-9:
                best_max = new_max
                best_routes = [route[:] for route in routes]
        else:
            no_improve += 1
            if no_improve >= 10:
                # restart
                routes, lengths = initial_solution()
                routes, lengths = local_search(routes, lengths)
                cur_max = max(lengths)
                if cur_max < best_max - 1e-9:
                    best_max = cur_max
                    best_routes = [route[:] for route in routes]
                no_improve = 0
    return best_routes