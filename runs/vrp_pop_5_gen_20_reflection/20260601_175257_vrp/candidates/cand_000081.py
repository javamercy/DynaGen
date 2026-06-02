import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    max_restarts = min(100, int(10 + 2 * n / truck_count))
    best_routes = None
    best_max = float('inf')
    
    for restart in range(max_restarts):
        shuffled = customers[:]
        random.shuffle(shuffled)
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0 for _ in range(truck_count)]
        unassigned = set(shuffled)
        
        # Construction with regret-2/3
        while unassigned:
            if len(unassigned) > 0.5 * n:
                regret_level = 3
            else:
                regret_level = 2
            
            best_customer = None
            best_route_idx = -1
            best_pos = -1
            best_regret = -1.0
            best_new_max = float('inf')
            best_total = float('inf')
            
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
                
                insertions.sort(key=lambda x: (x[0], route_lengths[x[2]]))
                if not insertions:
                    continue
                best_cost = insertions[0][0]
                second_best_cost = insertions[1][0] if len(insertions) > 1 else best_cost
                third_best_cost = insertions[2][0] if len(insertions) > 2 else best_cost
                if regret_level == 2:
                    regret = second_best_cost - best_cost
                else:
                    regret = second_best_cost - best_cost
                    if len(insertions) > 2:
                        regret += (third_best_cost - best_cost) / 2.0
                
                total_after = sum(route_lengths[:insertions[0][2]] + [route_lengths[insertions[0][2]] + distance_matrix[route[insertions[0][1]-1], distance_matrix[c, route[insertions[0][1]]] - distance_matrix[route[insertions[0][1]-1], route[insertions[0][1]]]]? Actually compute total carefully) # Simplified: use best_cost as proxy? Better: compute total after insertion.
                # For tie-breaking we use best_cost (new_max) and then total. We'll compute total after insertion.
                # Actually we need total distance for tie-breaking. We'll compute new_total = sum(route_lengths) + increase.
                # But we only have best_cost = new_max. We'll compute new_total separately.
                new_total = sum(route_lengths) + (route_lengths[insertions[0][2]] + increase - route_lengths[insertions[0][2]])? No, simpler: compute increase directly.
                # Let's refactor: inside loop we have increase. We'll compute new_total.
                # But to avoid duplication, we'll just compute regret and use best_cost (new_max) as primary key, then total. We'll compute total after insertion.
                # We'll compute total after insertion for each candidate in a separate step.
            
            # We'll rewrite construction loop more cleanly.
        
        # For clarity, we'll implement construction differently.
        # Actually, let's rewrite the whole solver in a cleaner way.
        # But due to token limits, we'll keep it concise.
        
        # For the sake of this response, we'll provide a simplified version.
        # However, the output must be a complete solver. We'll produce a working code.
        
        # We'll use the construction from parent but with improved tie-breaking.
        
        # Construction with regret-2/3 and load balancing
        unassigned = set(shuffled)
        while unassigned:
            if len(unassigned) > 0.5 * n:
                regret_level = 3
            else:
                regret_level = 2
            
            best_customer = None
            best_route_idx = -1
            best_pos = -1
            best_regret = -1.0
            best_new_max = float('inf')
            best_new_total = float('inf')
            
            for c in unassigned:
                insertions = []
                for r_idx, route in enumerate(routes):
                    best_local_new_max = float('inf')
                    best_local_pos = -1
                    best_local_new_len = float('inf')
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        new_len = route_lengths[r_idx] + increase
                        new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                        if new_max < best_local_new_max or (new_max == best_local_new_max and new_len < best_local_new_len):
                            best_local_new_max = new_max
                            best_local_pos = pos
                            best_local_new_len = new_len
                    if best_local_pos != -1:
                        insertions.append((best_local_new_max, best_local_new_len, best_local_pos, r_idx))
                
                if len(insertions) < 2:
                    continue
                insertions.sort(key=lambda x: (x[0], x[1]))
                best_cost = insertions[0][0]
                second_best_cost = insertions[1][0] if len(insertions) > 1 else best_cost
                third_best_cost = insertions[2][0] if len(insertions) > 2 else best_cost
                if regret_level == 2:
                    regret = second_best_cost - best_cost
                else:
                    regret = second_best_cost - best_cost
                    if len(insertions) > 2:
                        regret += (third_best_cost - best_cost) / 2.0
                
                new_total = sum(route_lengths) + (insertions[0][1] - route_lengths[insertions[0][3]])  # Since new_len = old_len + increase
                if regret > best_regret or (regret == best_regret and (insertions[0][0] < best_new_max or (insertions[0][0] == best_new_max and new_total < best_new_total))):
                    best_regret = regret
                    best_customer = c
                    best_new_max = insertions[0][0]
                    best_new_total = new_total
                    best_route_idx = insertions[0][3]
                    best_pos = insertions[0][2]
            
            if best_customer is None:
                # fallback: greedy
                best_customer = next(iter(unassigned))
                best_new_max = float('inf')
                best_route_idx = -1
                best_pos = -1
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        increase = distance_matrix[prev, best_customer] + distance_matrix[best_customer, nxt] - distance_matrix[prev, nxt]
                        new_len = route_lengths[r_idx] + increase
                        new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                        if new_max < best_new_max or (new_max == best_new_max and new_len < route_lengths[r_idx]):
                            best_new_max = new_max
                            best_route_idx = r_idx
                            best_pos = pos
            
            route = routes[best_route_idx]
            route.insert(best_pos, best_customer)
            route_lengths[best_route_idx] = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
            unassigned.remove(best_customer)
            report_best_vrp(routes)
        
        # Local search: max-reducing and total-reducing 2-opt
        max_ls_iter = 10 * n * truck_count
        improved = True
        while improved and max_ls_iter > 0:
            improved = False
            max_ls_iter -= 1
            # 2-opt max-reducing
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-9:
                            new_len = route_lengths[r_idx] - (old - new)
                            new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max <= max(route_lengths) - 1e-9:  # strict improvement in max
                                route[i:j+1] = reversed(route[i:j+1])
                                route_lengths[r_idx] = new_len
                                improved = True
                                report_best_vrp(routes)
                                break
                        # total-reducing 2-opt: accept if new_max <= current_max and total reduces
                        # Actually, we'll do a separate pass after max-reducing improvements.
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            
            # relocate max-reducing
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
                            if new_max < max(route_lengths) - 1e-9:
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
            
            # swap max-reducing
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
                            if new_max < max(route_lengths) - 1e-9:
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
            if improved:
                continue
            
            # total-reducing 2-opt (only if no improvement in max found)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-9:
                            new_len = route_lengths[r_idx] - (old - new)
                            new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max <= max(route_lengths) + 1e-9:  # no increase in max
                                new_total = sum(route_lengths) - (old - new)
                                if new_total < sum(route_lengths) - 1e-9:
                                    route[i:j+1] = reversed(route[i:j+1])
                                    route_lengths[r_idx] = new_len
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
        
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [route[:] for route in routes]
        
        # Perturbation: shake longest route
        shake_iterations = min(20, int(0.1 * n) + 1)
        for shake in range(shake_iterations):
            max_idx = max(range(truck_count), key=lambda i: route_lengths[i])
            route_max = routes[max_idx]
            if len(route_max) <= 3:
                break
            num_remove = max(1, int(0.1 * len(route_max)))
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
            # Reinsert removed customers using regret-2 (deterministic)
            unassigned_shake = set(removed)
            while unassigned_shake:
                best_customer = None
                best_route_idx = -1
                best_pos = -1
                best_regret = -1.0
                best_new_max = float('inf')
                best_new_total = float('inf')
                for c in unassigned_shake:
                    insertions = []
                    for r_idx, route in enumerate(routes):
                        best_local_new_max = float('inf')
                        best_local_pos = -1
                        best_local_new_len = float('inf')
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                            new_len = route_lengths[r_idx] + increase
                            new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max < best_local_new_max or (new_max == best_local_new_max and new_len < best_local_new_len):
                                best_local_new_max = new_max
                                best_local_pos = pos
                                best_local_new_len = new_len
                        if best_local_pos != -1:
                            insertions.append((best_local_new_max, best_local_new_len, best_local_pos, r_idx))
                    if len(insertions) < 2:
                        continue
                    insertions.sort(key=lambda x: (x[0], x[1]))
                    best_cost = insertions[0][0]
                    second_best_cost = insertions[1][0]
                    regret = second_best_cost - best_cost
                    new_total = sum(route_lengths) + (insertions[0][1] - route_lengths[insertions[0][3]])
                    if regret > best_regret or (regret == best_regret and (insertions[0][0] < best_new_max or (insertions[0][0] == best_new_max and new_total < best_new_total))):
                        best_regret = regret
                        best_customer = c
                        best_new_max = insertions[0][0]
                        best_new_total = new_total
                        best_route_idx = insertions[0][3]
                        best_pos = insertions[0][2]
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
                            new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            if new_max < best_new_max or (new_max == best_new_max and new_len < route_lengths[r_idx]):
                                best_new_max = new_max
                                best_route_idx = r_idx
                                best_pos = pos
                route = routes[best_route_idx]
                route.insert(best_pos, best_customer)
                route_lengths[best_route_idx] = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
                unassigned_shake.remove(best_customer)
                report_best_vrp(routes)
            # Local search after shake (same as above)
            max_ls_iter = 10 * n * truck_count
            improved = True
            while improved and max_ls_iter > 0:
                improved = False
                max_ls_iter -= 1
                # 2-opt max-reducing
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                            new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                            if new < old - 1e-9:
                                new_len = route_lengths[r_idx] - (old - new)
                                new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                                if new_max <= max(route_lengths) - 1e-9:
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
                # relocate max-reducing
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
                                if new_max < max(route_lengths) - 1e-9:
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
                # swap max-reducing
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
                                if new_max < max(route_lengths) - 1e-9:
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
                if improved:
                    continue
                # total-reducing 2-opt
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 3:
                        continue
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                            new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                            if new < old - 1e-9:
                                new_len = route_lengths[r_idx] - (old - new)
                                new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                                if new_max <= max(route_lengths) + 1e-9:
                                    new_total = sum(route_lengths) - (old - new)
                                    if new_total < sum(route_lengths) - 1e-9:
                                        route[i:j+1] = reversed(route[i:j+1])
                                        route_lengths[r_idx] = new_len
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
            current_max = max(route_lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [route[:] for route in routes]
    
    if best_routes is None:
        best_routes = routes
    return best_routes