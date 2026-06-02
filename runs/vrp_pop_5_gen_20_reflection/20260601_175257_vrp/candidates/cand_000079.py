import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    max_restarts = min(30, 10 + n // truck_count)
    best_routes = None
    best_max = float('inf')
    
    for restart in range(max_restarts):
        shuffled = customers[:]
        random.shuffle(shuffled)
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0 for _ in range(truck_count)]
        unassigned = set(shuffled)
        
        # Construction: regret-2 with tie-breaking
        while unassigned:
            best_customer = None
            best_route_idx = -1
            best_pos = -1
            best_regret = -1.0
            best_new_max = float('inf')
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
                        if new_max < best_local_cost or (new_max == best_local_cost and new_len < route_lengths[r_idx]):
                            best_local_cost = new_max
                            best_local_pos = pos
                    insertions.append((best_local_cost, best_local_pos, r_idx))
                insertions.sort(key=lambda x: (x[0], route_lengths[x[2]], x[1]))
                if len(insertions) < 2:
                    continue
                best_cost = insertions[0][0]
                second_best_cost = insertions[1][0]
                regret = second_best_cost - best_cost
                if regret > best_regret or (regret == best_regret and (best_cost < best_new_max or (best_cost == best_new_max and route_lengths[insertions[0][2]] < best_route_len))):
                    best_regret = regret
                    best_customer = c
                    best_new_max = best_cost
                    best_route_idx = insertions[0][2]
                    best_pos = insertions[0][1]
                    best_route_len = route_lengths[best_route_idx]
            if best_customer is None:
                # fallback
                for c in unassigned:
                    best_c = c
                    break
                best_new_max = float('inf')
                best_route_idx = -1
                best_pos = -1
                best_route_len = float('inf')
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        increase = distance_matrix[prev, best_c] + distance_matrix[best_c, nxt] - distance_matrix[prev, nxt]
                        new_len = route_lengths[r_idx] + increase
                        new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                        if new_max < best_new_max or (new_max == best_new_max and new_len < best_route_len):
                            best_new_max = new_max
                            best_route_idx = r_idx
                            best_pos = pos
                            best_route_len = new_len
                best_customer = best_c
            route = routes[best_route_idx]
            route.insert(best_pos, best_customer)
            route_lengths[best_route_idx] = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
            unassigned.remove(best_customer)
            report_best_vrp(routes)
        
        # Local search
        def local_search(routes, route_lengths):
            max_iter = 3 * n * truck_count + 10
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
                            new_dist = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                            if new_dist < old - 1e-9:
                                route[i:j+1] = reversed(route[i:j+1])
                                route_lengths[r_idx] -= old - new_dist
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
        
        local_search(routes, route_lengths)
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [route[:] for route in routes]
        
        # Shake phase - adaptive removal percentage
        shake_iters = min(20, int(0.3 * n) + 1)
        for shake in range(shake_iters):
            # find longest route
            max_idx = max(range(truck_count), key=lambda i: route_lengths[i])
            route_max = routes[max_idx]
            if len(route_max) <= 3:
                break
            # remove adaptive percentage (20-25%) of customers from longest route
            remove_frac = random.uniform(0.2, 0.25)
            num_remove = max(1, int(remove_frac * len(route_max)))
            if len(route_max) - 2 <= num_remove:
                num_remove = len(route_max) - 3
            if num_remove <= 0:
                break
            removed = []
            for _ in range(num_remove):
                if len(route_max) <= 2:
                    break
                idx = random.randint(1, len(route_max)-2)
                c = route_max[idx]
                removed.append(c)
                prev = route_max[idx-1]
                nxt = route_max[idx+1]
                cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                route_lengths[max_idx] -= cost_remove
                route_max.pop(idx)
            # reinsert with regret-2
            unassigned_shake = set(removed)
            while unassigned_shake:
                best_customer = None
                best_route_idx = -1
                best_pos = -1
                best_regret = -1.0
                best_new_max = float('inf')
                best_route_len = float('inf')
                for c in unassigned_shake:
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
                            if new_max < best_local_cost or (new_max == best_local_cost and new_len < route_lengths[r_idx]):
                                best_local_cost = new_max
                                best_local_pos = pos
                        insertions.append((best_local_cost, best_local_pos, r_idx))
                    insertions.sort(key=lambda x: (x[0], route_lengths[x[2]], x[1]))
                    if len(insertions) < 2:
                        continue
                    best_cost = insertions[0][0]
                    second_best_cost = insertions[1][0]
                    regret = second_best_cost - best_cost
                    if regret > best_regret or (regret == best_regret and (best_cost < best_new_max or (best_cost == best_new_max and route_lengths[insertions[0][2]] < best_route_len))):
                        best_regret = regret
                        best_customer = c
                        best_new_max = best_cost
                        best_route_idx = insertions[0][2]
                        best_pos = insertions[0][1]
                        best_route_len = route_lengths[best_route_idx]
                if best_customer is None:
                    # fallback
                    for c in unassigned_shake:
                        best_c = c
                        break
                    best_new_max = float('inf')
                    best_route_idx = -1
                    best_pos = -1
                    best_route_len = float('inf')
                    for r_idx, route in enumerate(routes):
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            increase = distance_matrix[prev, best_c] + distance_matrix[best_c, nxt] - distance_matrix[prev, nxt]
                            new_len = route_lengths[r_idx] + increase
                            new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max < best_new_max or (new_max == best_new_max and new_len < best_route_len):
                                best_new_max = new_max
                                best_route_idx = r_idx
                                best_pos = pos
                                best_route_len = new_len
                    best_customer = best_c
                route = routes[best_route_idx]
                route.insert(best_pos, best_customer)
                route_lengths[best_route_idx] = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
                unassigned_shake.remove(best_customer)
                report_best_vrp(routes)
            # local search after shake
            local_search(routes, route_lengths)
            current_max = max(route_lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [route[:] for route in routes]
    
    if best_routes is None:
        best_routes = routes
    return best_routes