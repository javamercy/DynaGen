import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    best_routes = None
    best_max = float('inf')
    
    max_restarts = max(1, min(100, int(10 + n / truck_count)))
    
    for restart in range(max_restarts):
        # Build initial solution with load-balanced regret-2
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0 for _ in range(truck_count)]
        unassigned = customers[:]
        random.shuffle(unassigned)
        
        while unassigned:
            best_customer = None
            best_regret = -1e9
            best_route_idx = -1
            best_pos = -1
            best_new_max = float('inf')
            best_route_length = float('inf')
            
            for c in unassigned:
                insertions = []
                for r_idx, route in enumerate(routes):
                    best_local_max = float('inf')
                    best_local_pos = -1
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        new_len = route_lengths[r_idx] + increase
                        new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                        if new_max < best_local_max or (new_max == best_local_max and route_lengths[r_idx] < best_route_length):
                            best_local_max = new_max
                            best_local_pos = pos
                    insertions.append((best_local_max, best_local_pos, r_idx))
                insertions.sort(key=lambda x: (x[0], route_lengths[x[2]]))
                if not insertions:
                    continue
                best_cost = insertions[0][0]
                second_best_cost = insertions[1][0] if len(insertions) > 1 else best_cost
                regret = second_best_cost - best_cost
                # Prefer customers with high regret, then lower best cost, then lower route length of target route
                if regret > best_regret or (regret == best_regret and (best_cost < best_new_max or (best_cost == best_new_max and route_lengths[insertions[0][2]] < best_route_length))):
                    best_regret = regret
                    best_customer = c
                    best_new_max = best_cost
                    best_route_idx = insertions[0][2]
                    best_pos = insertions[0][1]
                    best_route_length = route_lengths[best_route_idx]
            
            if best_customer is None:
                # Fallback: assign to route with smallest load
                best_customer = unassigned[0]
                best_new_max = float('inf')
                best_route_idx = -1
                best_pos = -1
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        increase = distance_matrix[prev, best_customer] + distance_matrix[best_customer, nxt] - distance_matrix[prev, nxt]
                        new_len = route_lengths[r_idx] + increase
                        new_max_ = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                        if new_max_ < best_new_max:
                            best_new_max = new_max_
                            best_route_idx = r_idx
                            best_pos = pos
            
            route = routes[best_route_idx]
            route.insert(best_pos, best_customer)
            route_lengths[best_route_idx] += distance_matrix[route[best_pos-1], best_customer] + distance_matrix[best_customer, route[best_pos+1]] - distance_matrix[route[best_pos-1], route[best_pos+1]]
            unassigned.remove(best_customer)
            report_best_vrp(routes)
        
        # Local search for max reduction
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
                            new_len = route_lengths[r_idx] - old + new
                            new_max_val = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max_val < max(route_lengths):
                                route[i:j+1] = reversed(route[i:j+1])
                                route_lengths[r_idx] = new_len
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
                            new_max_val = max(route_lengths[:r_from] + [new_len_from] + route_lengths[r_from+1:r_to] + [new_len_to] + route_lengths[r_to+1:])
                            if new_max_val < max(route_lengths) - 1e-9:
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
                            new_max_val = max(route_lengths[:r1] + [new_len1] + route_lengths[r1+1:r2] + [new_len2] + route_lengths[r2+1:])
                            if new_max_val < max(route_lengths) - 1e-9:
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
        
        # Intra-route 2-opt for total distance improvement (no worsen max)
        total_iter = 5 * n * truck_count + 10
        total_improved = True
        while total_improved and total_iter > 0:
            total_improved = False
            total_iter -= 1
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-9:
                            new_len = route_lengths[r_idx] - old + new
                            new_max_val = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max_val <= max(route_lengths) and new_len < route_lengths[r_idx]:
                                route[i:j+1] = reversed(route[i:j+1])
                                route_lengths[r_idx] = new_len
                                total_improved = True
                                report_best_vrp(routes)
                                break
                    if total_improved:
                        break
                if total_improved:
                    break
        
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [route[:] for route in routes]
        
        # Shake phase
        shake_iters = min(15, max(1, int(0.1 * n)))
        for shake in range(shake_iters):
            # Choose route to remove from
            route_indices = [i for i in range(truck_count) if len(routes[i]) > 2]
            if not route_indices:
                break
            if random.random() < 0.5:
                max_idx = max(route_indices, key=lambda i: route_lengths[i])
            else:
                max_idx = random.choice(route_indices)
            route_max = routes[max_idx]
            num_remove = max(1, int(0.1 * n))
            if len(route_max) - 2 <= num_remove:
                num_remove = len(route_max) - 3
            if num_remove <= 0:
                break
            removed = []
            indices = list(range(1, len(route_max)-1))
            random.shuffle(indices)
            indices_to_remove = sorted(indices[:num_remove], reverse=True)
            for idx in indices_to_remove:
                c = route_max[idx]
                removed.append(c)
                prev = route_max[idx-1]
                nxt = route_max[idx+1]
                cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                route_lengths[max_idx] -= cost_remove
                del route_max[idx]
            unassigned_shake = set(removed)
            # Regret-2 insertion (minimizing max distance)
            while unassigned_shake:
                best_customer = None
                best_regret = -1e9
                best_new_max = float('inf')
                best_route_idx = -1
                best_pos = -1
                for c in unassigned_shake:
                    insertions = []
                    for r_idx, route in enumerate(routes):
                        best_local_new_max = float('inf')
                        best_local_pos = -1
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                            new_len = route_lengths[r_idx] + increase
                            new_max_val = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max_val < best_local_new_max:
                                best_local_new_max = new_max_val
                                best_local_pos = pos
                        insertions.append((best_local_new_max, best_local_pos, r_idx))
                    insertions.sort(key=lambda x: x[0])
                    if len(insertions) >= 2:
                        regret = insertions[1][0] - insertions[0][0]
                    else:
                        regret = 0
                    if regret > best_regret or (regret == best_regret and (insertions[0][0] < best_new_max or (insertions[0][0] == best_new_max and route_lengths[insertions[0][2]] < route_lengths[best_route_idx] if best_route_idx != -1 else False))):
                        best_regret = regret
                        best_customer = c
                        best_new_max = insertions[0][0]
                        best_route_idx = insertions[0][2]
                        best_pos = insertions[0][1]
                if best_customer is None:
                    best_customer = next(iter(unassigned_shake))
                    best_new_max = float('inf')
                    best_route_idx = -1
                    best_pos = -1
                    for r_idx, route in enumerate(routes):
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            increase = distance_matrix[prev, best_customer] + distance_matrix[best_customer, nxt] - distance_matrix[prev, nxt]
                            new_len = route_lengths[r_idx] + increase
                            new_max_val = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max_val < best_new_max:
                                best_new_max = new_max_val
                                best_route_idx = r_idx
                                best_pos = pos
                route = routes[best_route_idx]
                route.insert(best_pos, best_customer)
                route_lengths[best_route_idx] += distance_matrix[route[best_pos-1], best_customer] + distance_matrix[best_customer, route[best_pos+1]] - distance_matrix[route[best_pos-1], route[best_pos+1]]
                unassigned_shake.remove(best_customer)
                report_best_vrp(routes)
            # Local search after shake (max-reducing)
            max_iter_ls = 5 * n * truck_count + 10
            improved_ls = True
            while improved_ls and max_iter_ls > 0:
                improved_ls = False
                max_iter_ls -= 1
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
                                new_len = route_lengths[r_idx] - old + new
                                new_max_val = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                                if new_max_val < max(route_lengths):
                                    route[i:j+1] = reversed(route[i:j+1])
                                    route_lengths[r_idx] = new_len
                                    improved_ls = True
                                    report_best_vrp(routes)
                                    break
                        if improved_ls:
                            break
                    if improved_ls:
                        break
                if improved_ls:
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
                                new_max_val = max(route_lengths[:r_from] + [new_len_from] + route_lengths[r_from+1:r_to] + [new_len_to] + route_lengths[r_to+1:])
                                if new_max_val < max(route_lengths) - 1e-9:
                                    route_from.pop(idx_c)
                                    route_lengths[r_from] = new_len_from
                                    route_to.insert(pos, c)
                                    route_lengths[r_to] = new_len_to
                                    improved_ls = True
                                    report_best_vrp(routes)
                                    break
                            if improved_ls:
                                break
                        if improved_ls:
                            break
                    if improved_ls:
                        break
                if improved_ls:
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
                                new_max_val = max(route_lengths[:r1] + [new_len1] + route_lengths[r1+1:r2] + [new_len2] + route_lengths[r2+1:])
                                if new_max_val < max(route_lengths) - 1e-9:
                                    del route1[idx1]
                                    del route2[idx2]
                                    route1.insert(idx1, c2)
                                    route2.insert(idx2, c1)
                                    route_lengths[r1] = new_len1
                                    route_lengths[r2] = new_len2
                                    improved_ls = True
                                    report_best_vrp(routes)
                                    break
                            if improved_ls:
                                break
                        if improved_ls:
                            break
                    if improved_ls:
                        break
            # Intra-route 2-opt after shake
            total_iter_shake = 5 * n * truck_count + 10
            total_improved_shake = True
            while total_improved_shake and total_iter_shake > 0:
                total_improved_shake = False
                total_iter_shake -= 1
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                            new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                            if new < old - 1e-9:
                                new_len = route_lengths[r_idx] - old + new
                                new_max_val = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                                if new_max_val <= max(route_lengths) and new_len < route_lengths[r_idx]:
                                    route[i:j+1] = reversed(route[i:j+1])
                                    route_lengths[r_idx] = new_len
                                    total_improved_shake = True
                                    report_best_vrp(routes)
                                    break
                        if total_improved_shake:
                            break
                    if total_improved_shake:
                        break
            current_max = max(route_lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [route[:] for route in routes]
    
    if best_routes is None:
        best_routes = routes
    return best_routes