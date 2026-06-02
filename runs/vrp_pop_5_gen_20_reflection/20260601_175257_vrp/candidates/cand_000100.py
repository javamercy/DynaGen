import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    max_restarts = min(100, int(10 + 2 * n / truck_count))
    best_routes = None
    best_max = float('inf')

    def regret_insertion(unassigned, routes, route_lengths):
        while unassigned:
            if len(unassigned) > 0.5 * (n - 1):
                regret_level = 3
            else:
                regret_level = 2
            best_customer = None
            best_route_idx = -1
            best_pos = -1
            best_regret = -1.0
            best_cost = float('inf')
            best_route_len = float('inf')
            for c in unassigned:
                insertions = []
                for r_idx, route in enumerate(routes):
                    best_local_cost = float('inf')
                    best_local_pos = -1
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        new_len = route_lengths[r_idx] + increase
                        new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                        if new_max < best_local_cost or (new_max == best_local_cost and route_lengths[r_idx] < best_route_len):
                            best_local_cost = new_max
                            best_local_pos = pos
                            best_local_route_len = route_lengths[r_idx]
                    insertions.append((best_local_cost, best_local_pos, r_idx, best_local_route_len))
                insertions.sort(key=lambda x: (x[0], x[3], x[1]))
                if len(insertions) < 2:
                    continue
                best = insertions[0][0]
                second = insertions[1][0]
                regret_2 = second - best
                if regret_level == 3 and len(insertions) > 2:
                    third = insertions[2][0]
                    regret_3 = (third - best) / 2.0
                else:
                    regret_3 = 0.0
                regret = regret_2 + regret_3
                if regret > best_regret or (regret == best_regret and (best < best_cost or (best == best_cost and insertions[0][3] < best_route_len))):
                    best_regret = regret
                    best_customer = c
                    best_route_idx = insertions[0][2]
                    best_pos = insertions[0][1]
                    best_cost = best
                    best_route_len = insertions[0][3]
            if best_customer is None:
                best_customer = next(iter(unassigned))
                best_cost = float('inf')
                best_route_idx = -1
                best_pos = -1
                best_route_len = float('inf')
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        increase = distance_matrix[prev, best_customer] + distance_matrix[best_customer, nxt] - distance_matrix[prev, nxt]
                        new_len = route_lengths[r_idx] + increase
                        new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                        if new_max < best_cost or (new_max == best_cost and route_lengths[r_idx] < best_route_len):
                            best_cost = new_max
                            best_route_idx = r_idx
                            best_pos = pos
                            best_route_len = route_lengths[r_idx]
            route = routes[best_route_idx]
            route.insert(best_pos, best_customer)
            route_lengths[best_route_idx] = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
            unassigned.remove(best_customer)
            report_best_vrp(routes)

    def local_search(routes, route_lengths):
        max_iter = 5 * n * truck_count + 10
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
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
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # relocate
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
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    for restart in range(max_restarts):
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0 for _ in range(truck_count)]
        unassigned = set(customers)
        regret_insertion(unassigned, routes, route_lengths)
        local_search(routes, route_lengths)
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [route[:] for route in routes]

        # Perturbation: ruin and recreate
        shake_iters = int(0.3 * n) + 5
        for _ in range(shake_iters):
            # Remove 30% random customers
            num_remove = max(1, int(0.3 * (n - 1)))
            all_custs = []
            for r in routes:
                for c in r[1:-1]:
                    all_custs.append(c)
            if num_remove >= len(all_custs):
                break
            removed = random.sample(all_custs, num_remove)
            # Remove from routes
            for r_idx in range(truck_count):
                route = routes[r_idx]
                i = 1
                while i < len(route) - 1:
                    if route[i] in removed:
                        c = route[i]
                        prev = route[i-1]
                        nxt = route[i+1]
                        cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        route_lengths[r_idx] -= cost_remove
                        route.pop(i)
                    else:
                        i += 1
            # Reinsert
            unassigned = set(removed)
            regret_insertion(unassigned, routes, route_lengths)
            # Short local search
            local_search(routes, route_lengths)
            current_max = max(route_lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [route[:] for route in routes]

    return best_routes