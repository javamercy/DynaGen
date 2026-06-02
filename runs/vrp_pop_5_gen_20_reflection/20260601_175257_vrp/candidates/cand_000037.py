import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    best_routes = None
    best_max = float('inf')
    max_restarts = min(100, int(10 + 2 * n / truck_count))
    for restart in range(max_restarts):
        shuffled = customers[:]
        random.shuffle(shuffled)
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0] * truck_count
        unassigned = set(shuffled)
        while unassigned:
            best_customer = None
            best_route_idx = -1
            best_pos = -1
            best_regret = -1.0
            best_new_max = float('inf')
            best_route_len = float('inf')
            for c in list(unassigned):
                insertions = []
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    local_best = float('inf')
                    local_pos = -1
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        delta = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        new_len = route_lengths[r_idx] + delta
                        new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                        if new_max < local_best:
                            local_best = new_max
                            local_pos = pos
                    if local_best < float('inf'):
                        insertions.append((local_best, r_idx, local_pos))
                insertions.sort(key=lambda x: (x[0], x[1]))
                if len(insertions) >= 2:
                    regret = insertions[1][0] - insertions[0][0]
                else:
                    regret = 0.0
                candidate_max = insertions[0][0] if insertions else float('inf')
                candidate_route_len = route_lengths[insertions[0][1]] if insertions else float('inf')
                if (regret > best_regret or
                    (regret == best_regret and (candidate_max < best_new_max or
                     (candidate_max == best_new_max and candidate_route_len < best_route_len)))):
                    best_regret = regret
                    best_customer = c
                    best_new_max = candidate_max
                    best_route_idx = insertions[0][1]
                    best_pos = insertions[0][2]
                    best_route_len = candidate_route_len
            route = routes[best_route_idx]
            route.insert(best_pos, best_customer)
            route_lengths[best_route_idx] = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
            unassigned.remove(best_customer)
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        # Local search (limited iterations)
        max_iter = 5 * n * truck_count + 50
        for _ in range(max_iter):
            improved = False
            # 2-opt
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
                            route_lengths[r_idx] -= old - new
                            improved = True
                            current_max = max(route_lengths)
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Relocate
            for r_from in range(truck_count):
                route_from = routes[r_from]
                if len(route_from) <= 2:
                    continue
                for idx_c in range(1, len(route_from)-1):
                    c = route_from[idx_c]
                    prev = route_from[idx_c-1]
                    nxt = route_from[idx_c+1]
                    cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                    new_len_from = route_lengths[r_from] - cost_remove
                    for r_to in range(truck_count):
                        if r_to == r_from:
                            continue
                        route_to = routes[r_to]
                        for pos in range(1, len(route_to)):
                            prev_to = route_to[pos-1]
                            nxt_to = route_to[pos]
                            cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                            new_len_to = route_lengths[r_to] + cost_insert
                            new_max = max(route_lengths[:r_from] + [new_len_from] + route_lengths[r_from+1:r_to] + [new_len_to] + route_lengths[r_to+1:])
                            current_max = max(route_lengths)
                            if new_max < current_max - 1e-9:
                                route_from.pop(idx_c)
                                route_lengths[r_from] = new_len_from
                                route_to.insert(pos, c)
                                route_lengths[r_to] = new_len_to
                                improved = True
                                current_max = new_max
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(best_routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Swap
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
                            new_len1 = route_lengths[r1] - cost_remove1 + cost_insert1
                            cost_insert2 = distance_matrix[prev2, c1] + distance_matrix[c1, nxt2] - distance_matrix[prev2, nxt2]
                            new_len2 = route_lengths[r2] - cost_remove2 + cost_insert2
                            new_max = max(route_lengths[:r1] + [new_len1] + route_lengths[r1+1:r2] + [new_len2] + route_lengths[r2+1:])
                            current_max = max(route_lengths)
                            if new_max < current_max - 1e-9:
                                del route1[idx1]
                                del route2[idx2]
                                route1.insert(idx1, c2)
                                route2.insert(idx2, c1)
                                route_lengths[r1] = new_len1
                                route_lengths[r2] = new_len2
                                improved = True
                                current_max = new_max
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(best_routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        # Perturbation and local search
        perturb_repeats = max(1, n // 20)
        for _ in range(perturb_repeats):
            non_empty = [i for i, r in enumerate(routes) if len(r) > 2]
            if not non_empty:
                break
            r_idx = random.choice(non_empty)
            route = routes[r_idx]
            idx = random.randint(1, len(route)-2)
            c = route[idx]
            prev = route[idx-1]
            nxt = route[idx+1]
            cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
            route.pop(idx)
            route_lengths[r_idx] -= cost_remove
            best_route = -1
            best_pos = -1
            best_new_max = max(route_lengths)
            for r_to in range(truck_count):
                rt = routes[r_to]
                for pos in range(1, len(rt)):
                    prev_to = rt[pos-1]
                    nxt_to = rt[pos]
                    cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                    new_len_to = route_lengths[r_to] + cost_insert
                    new_max = max(route_lengths[:r_to] + [new_len_to] + route_lengths[r_to+1:])
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_route = r_to
                        best_pos = pos
            if best_route != -1:
                routes[best_route].insert(best_pos, c)
                route_lengths[best_route] = sum(distance_matrix[routes[best_route][i], routes[best_route][i+1]] for i in range(len(routes[best_route])-1))
            else:
                route.insert(idx, c)
                route_lengths[r_idx] += cost_remove
            # local search again (same as above but with reduced iterations)
            for _ in range(10):
                improved = False
                # 2-opt
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
                                route_lengths[r_idx] -= old - new
                                improved = True
                                current_max = max(route_lengths)
                                if current_max < best_max:
                                    best_max = current_max
                                    best_routes = [r[:] for r in routes]
                                    report_best_vrp(best_routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
                # Relocate
                for r_from in range(truck_count):
                    route_from = routes[r_from]
                    if len(route_from) <= 2:
                        continue
                    for idx_c in range(1, len(route_from)-1):
                        c = route_from[idx_c]
                        prev = route_from[idx_c-1]
                        nxt = route_from[idx_c+1]
                        cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        new_len_from = route_lengths[r_from] - cost_remove
                        for r_to in range(truck_count):
                            if r_to == r_from:
                                continue
                            route_to = routes[r_to]
                            for pos in range(1, len(route_to)):
                                prev_to = route_to[pos-1]
                                nxt_to = route_to[pos]
                                cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                                new_len_to = route_lengths[r_to] + cost_insert
                                new_max = max(route_lengths[:r_from] + [new_len_from] + route_lengths[r_from+1:r_to] + [new_len_to] + route_lengths[r_to+1:])
                                current_max = max(route_lengths)
                                if new_max < current_max - 1e-9:
                                    route_from.pop(idx_c)
                                    route_lengths[r_from] = new_len_from
                                    route_to.insert(pos, c)
                                    route_lengths[r_to] = new_len_to
                                    improved = True
                                    current_max = new_max
                                    if current_max < best_max:
                                        best_max = current_max
                                        best_routes = [r[:] for r in routes]
                                        report_best_vrp(best_routes)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
                # Swap
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
                                new_len1 = route_lengths[r1] - cost_remove1 + cost_insert1
                                cost_insert2 = distance_matrix[prev2, c1] + distance_matrix[c1, nxt2] - distance_matrix[prev2, nxt2]
                                new_len2 = route_lengths[r2] - cost_remove2 + cost_insert2
                                new_max = max(route_lengths[:r1] + [new_len1] + route_lengths[r1+1:r2] + [new_len2] + route_lengths[r2+1:])
                                current_max = max(route_lengths)
                                if new_max < current_max - 1e-9:
                                    del route1[idx1]
                                    del route2[idx2]
                                    route1.insert(idx1, c2)
                                    route2.insert(idx2, c1)
                                    route_lengths[r1] = new_len1
                                    route_lengths[r2] = new_len2
                                    improved = True
                                    current_max = new_max
                                    if current_max < best_max:
                                        best_max = current_max
                                        best_routes = [r[:] for r in routes]
                                        report_best_vrp(best_routes)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break
            current_max = max(route_lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
    return best_routes