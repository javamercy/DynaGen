import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    best_routes = None
    best_max = float('inf')
    
    max_restarts = 5
    for restart in range(max_restarts):
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0] * truck_count
        unassigned = set(customers)
        
        # Regret-2 construction
        while unassigned:
            best_customer = None
            best_route_idx = -1
            best_pos = -1
            best_regret = -1.0
            best_cost = float('inf')
            best_len = float('inf')
            for c in unassigned:
                insertions = []
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        new_len = route_lengths[r_idx] + increase
                        new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                        insertions.append((new_max, new_len, r_idx, pos))
                insertions.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
                if len(insertions) >= 2:
                    cost1, len1, r1, p1 = insertions[0]
                    cost2, _, _, _ = insertions[1]
                    regret = cost2 - cost1
                    if regret > best_regret or (regret == best_regret and (cost1 < best_cost or (cost1 == best_cost and len1 < best_len))):
                        best_regret = regret
                        best_customer = c
                        best_route_idx = r1
                        best_pos = p1
                        best_cost = cost1
                        best_len = len1
            if best_customer is None:
                # fallback: pick first unassigned
                best_customer = next(iter(unassigned))
                best_route_idx = 0
                best_pos = 1
                best_cost = float('inf')
                best_len = float('inf')
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        increase = distance_matrix[prev, best_customer] + distance_matrix[best_customer, nxt] - distance_matrix[prev, nxt]
                        new_len = route_lengths[r_idx] + increase
                        new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                        if new_max < best_cost or (new_max == best_cost and new_len < best_len):
                            best_cost = new_max
                            best_len = new_len
                            best_route_idx = r_idx
                            best_pos = pos
            route = routes[best_route_idx]
            route.insert(best_pos, best_customer)
            route_lengths[best_route_idx] = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
            unassigned.remove(best_customer)
        
        # Local search (2-opt and relocate)
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
                        new_dist = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new_dist < old - 1e-9:
                            route[i:j+1] = reversed(route[i:j+1])
                            route_lengths[r_idx] -= old - new_dist
                            improved = True
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
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [route[:] for route in routes]
            report_best_vrp(best_routes)
        
        # Shake phase
        shake_iters = min(10, int(0.2 * n) + 1)
        for shake in range(shake_iters):
            max_idx = max(range(truck_count), key=lambda i: route_lengths[i])
            route_max = routes[max_idx]
            if len(route_max) <= 3:
                break
            num_remove = max(1, int(0.2 * n))
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
            # Reinsert removed customers with regret-2
            unassigned_shake = set(removed)
            while unassigned_shake:
                best_customer = None
                best_route_idx = -1
                best_pos = -1
                best_regret = -1.0
                best_cost = float('inf')
                best_len = float('inf')
                for c in unassigned_shake:
                    insertions = []
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                            new_len = route_lengths[r_idx] + increase
                            new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            insertions.append((new_max, new_len, r_idx, pos))
                    insertions.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
                    if len(insertions) >= 2:
                        cost1, len1, r1, p1 = insertions[0]
                        cost2, _, _, _ = insertions[1]
                        regret = cost2 - cost1
                        if regret > best_regret or (regret == best_regret and (cost1 < best_cost or (cost1 == best_cost and len1 < best_len))):
                            best_regret = regret
                            best_customer = c
                            best_route_idx = r1
                            best_pos = p1
                            best_cost = cost1
                            best_len = len1
                if best_customer is None:
                    best_customer = next(iter(unassigned_shake))
                    best_route_idx = 0
                    best_pos = 1
                    best_cost = float('inf')
                    best_len = float('inf')
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            increase = distance_matrix[prev, best_customer] + distance_matrix[best_customer, nxt] - distance_matrix[prev, nxt]
                            new_len = route_lengths[r_idx] + increase
                            new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max < best_cost or (new_max == best_cost and new_len < best_len):
                                best_cost = new_max
                                best_len = new_len
                                best_route_idx = r_idx
                                best_pos = pos
                route = routes[best_route_idx]
                route.insert(best_pos, best_customer)
                route_lengths[best_route_idx] = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
                unassigned_shake.remove(best_customer)
            # Local search after shake
            max_iter2 = 5 * n * truck_count + 10
            improved = True
            while improved and max_iter2 > 0:
                improved = False
                max_iter2 -= 1
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
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            current_max = max(route_lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [route[:] for route in routes]
                report_best_vrp(best_routes)
    
    if best_routes is None:
        best_routes = routes
    return best_routes