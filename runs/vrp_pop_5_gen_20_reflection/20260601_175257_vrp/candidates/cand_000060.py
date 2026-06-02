import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    max_restarts = min(100, max(10, int(10 + 2 * n / truck_count)))
    best_routes = None
    best_max = float('inf')
    
    def route_length(route):
        if len(route) < 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    for _ in range(max_restarts):
        # random initial assignment
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0 for _ in range(truck_count)]
        unassigned = set(customers)
        while unassigned:
            best_cust = None
            best_route = -1
            best_pos = -1
            best_regret = -1.0
            best_incumbent = float('inf')
            for c in unassigned:
                insertions = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        new_len = lengths[r_idx] + increase
                        new_max = max(lengths[:r_idx] + [new_len] + lengths[r_idx+1:])
                        insertions.append((new_max, r_idx, pos))
                insertions.sort(key=lambda x: (x[0], lengths[x[1]], x[2]))
                if len(insertions) == 0:
                    continue
                best = insertions[0]
                second = insertions[1] if len(insertions) > 1 else best
                third = insertions[2] if len(insertions) > 2 else best
                regret = (second[0] - best[0]) + (third[0] - best[0]) / 2.0
                if regret > best_regret or (regret == best_regret and best[0] < best_incumbent):
                    best_regret = regret
                    best_cust = c
                    best_route = best[1]
                    best_pos = best[2]
                    best_incumbent = best[0]
            if best_cust is None:
                # fallback: first in unassigned
                best_cust = next(iter(unassigned))
                best_incumbent = float('inf')
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        increase = distance_matrix[prev, best_cust] + distance_matrix[best_cust, nxt] - distance_matrix[prev, nxt]
                        new_len = lengths[r_idx] + increase
                        new_max = max(lengths[:r_idx] + [new_len] + lengths[r_idx+1:])
                        if new_max < best_incumbent:
                            best_incumbent = new_max
                            best_route = r_idx
                            best_pos = pos
            route = routes[best_route]
            route.insert(best_pos, best_cust)
            lengths[best_route] = route_length(route)
            unassigned.remove(best_cust)
            report_best_vrp(routes)
        
        # local search
        improvable = True
        max_iters = 5 * n * truck_count
        while improvable and max_iters > 0:
            improvable = False
            max_iters -= 1
            # intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-9:
                            route[i:j+1] = reversed(route[i:j+1])
                            lengths[r_idx] -= old - new
                            improvable = True
                            report_best_vrp(routes)
                            break
                    if improvable:
                        break
                if improvable:
                    break
            if improvable:
                continue
            # inter-route relocate
            for r_from in range(truck_count):
                route_from = routes[r_from]
                if len(route_from) <= 2:
                    continue
                for idx_c in range(1, len(route_from)-1):
                    c = route_from[idx_c]
                    prev = route_from[idx_c-1]
                    nxt = route_from[idx_c+1]
                    cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                    new_len_from = lengths[r_from] - cost_remove
                    for r_to in range(truck_count):
                        if r_to == r_from:
                            continue
                        route_to = routes[r_to]
                        for pos in range(1, len(route_to)):
                            prev_to = route_to[pos-1]
                            nxt_to = route_to[pos]
                            cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                            new_len_to = lengths[r_to] + cost_insert
                            new_max = max(lengths[:r_from] + [new_len_from] + lengths[r_from+1:r_to] + [new_len_to] + lengths[r_to+1:])
                            current_max = max(lengths)
                            if new_max < current_max - 1e-9:
                                route_from.pop(idx_c)
                                lengths[r_from] = new_len_from
                                route_to.insert(pos, c)
                                lengths[r_to] = new_len_to
                                improvable = True
                                report_best_vrp(routes)
                                break
                        if improvable:
                            break
                    if improvable:
                        break
                if improvable:
                    break
            if improvable:
                continue
            # inter-route swap
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
                            new_max = max(lengths[:r1] + [new_len1] + lengths[r1+1:r2] + [new_len2] + lengths[r2+1:])
                            current_max = max(lengths)
                            if new_max < current_max - 1e-9:
                                del route1[idx1]
                                del route2[idx2]
                                route1.insert(idx1, c2)
                                route2.insert(idx2, c1)
                                lengths[r1] = new_len1
                                lengths[r2] = new_len2
                                improvable = True
                                report_best_vrp(routes)
                                break
                        if improvable:
                            break
                    if improvable:
                        break
                if improvable:
                    break
        
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [route[:] for route in routes]
        
        # shake: remove a random subset and reinsert with regret
        shake_iters = max(1, int(0.2 * n / truck_count))
        for _ in range(shake_iters):
            num_remove = max(1, int(random.uniform(0.1, 0.2) * (n - 1)))
            all_cust = [cust for route in routes for cust in route[1:-1]]
            if len(all_cust) < num_remove:
                num_remove = len(all_cust)
            removed = random.sample(all_cust, num_remove)
            for c in removed:
                for r_idx, route in enumerate(routes):
                    if c in route:
                        idx = route.index(c)
                        prev = route[idx-1]
                        nxt = route[idx+1]
                        cost = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        route.pop(idx)
                        lengths[r_idx] -= cost
                        break
            unassigned = set(removed)
            while unassigned:
                best_cust = None
                best_route = -1
                best_pos = -1
                best_regret = -1.0
                best_incumbent = float('inf')
                for c in unassigned:
                    insertions = []
                    for r_idx, route in enumerate(routes):
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                            new_len = lengths[r_idx] + increase
                            new_max = max(lengths[:r_idx] + [new_len] + lengths[r_idx+1:])
                            insertions.append((new_max, r_idx, pos))
                    insertions.sort(key=lambda x: (x[0], lengths[x[1]], x[2]))
                    if len(insertions) == 0:
                        continue
                    best = insertions[0]
                    second = insertions[1] if len(insertions) > 1 else best
                    third = insertions[2] if len(insertions) > 2 else best
                    regret = (second[0] - best[0]) + (third[0] - best[0]) / 2.0
                    if regret > best_regret or (regret == best_regret and best[0] < best_incumbent):
                        best_regret = regret
                        best_cust = c
                        best_route = best[1]
                        best_pos = best[2]
                        best_incumbent = best[0]
                if best_cust is None:
                    best_cust = next(iter(unassigned))
                    for r_idx, route in enumerate(routes):
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            increase = distance_matrix[prev, best_cust] + distance_matrix[best_cust, nxt] - distance_matrix[prev, nxt]
                            new_len = lengths[r_idx] + increase
                            new_max = max(lengths[:r_idx] + [new_len] + lengths[r_idx+1:])
                            if new_max < best_incumbent:
                                best_incumbent = new_max
                                best_route = r_idx
                                best_pos = pos
                route = routes[best_route]
                route.insert(best_pos, best_cust)
                lengths[best_route] = route_length(route)
                unassigned.remove(best_cust)
                report_best_vrp(routes)
            # local search after shake
            improvable = True
            max_iters2 = 3 * n * truck_count
            while improvable and max_iters2 > 0:
                improvable = False
                max_iters2 -= 1
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                            new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                            if new < old - 1e-9:
                                route[i:j+1] = reversed(route[i:j+1])
                                lengths[r_idx] -= old - new
                                improvable = True
                                report_best_vrp(routes)
                                break
                        if improvable:
                            break
                    if improvable:
                        break
                if improvable:
                    continue
                for r_from in range(truck_count):
                    route_from = routes[r_from]
                    if len(route_from) <= 2:
                        continue
                    for idx_c in range(1, len(route_from)-1):
                        c = route_from[idx_c]
                        prev = route_from[idx_c-1]
                        nxt = route_from[idx_c+1]
                        cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        new_len_from = lengths[r_from] - cost_remove
                        for r_to in range(truck_count):
                            if r_to == r_from:
                                continue
                            route_to = routes[r_to]
                            for pos in range(1, len(route_to)):
                                prev_to = route_to[pos-1]
                                nxt_to = route_to[pos]
                                cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                                new_len_to = lengths[r_to] + cost_insert
                                new_max = max(lengths[:r_from] + [new_len_from] + lengths[r_from+1:r_to] + [new_len_to] + lengths[r_to+1:])
                                current_max = max(lengths)
                                if new_max < current_max - 1e-9:
                                    route_from.pop(idx_c)
                                    lengths[r_from] = new_len_from
                                    route_to.insert(pos, c)
                                    lengths[r_to] = new_len_to
                                    improvable = True
                                    report_best_vrp(routes)
                                    break
                            if improvable:
                                break
                        if improvable:
                            break
                    if improvable:
                        break
                if improvable:
                    continue
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
                                new_max = max(lengths[:r1] + [new_len1] + lengths[r1+1:r2] + [new_len2] + lengths[r2+1:])
                                current_max = max(lengths)
                                if new_max < current_max - 1e-9:
                                    del route1[idx1]
                                    del route2[idx2]
                                    route1.insert(idx1, c2)
                                    route2.insert(idx2, c1)
                                    lengths[r1] = new_len1
                                    lengths[r2] = new_len2
                                    improvable = True
                                    report_best_vrp(routes)
                                    break
                            if improvable:
                                break
                        if improvable:
                            break
                    if improvable:
                        break
            current_max = max(lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [route[:] for route in routes]
    
    if best_routes is None:
        best_routes = routes
    return best_routes