import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    max_restarts = min(100, max(10, int(10 + 2 * n / truck_count)))
    best_routes = None
    best_max = float('inf')
    
    def compute_route_length(route):
        if len(route) < 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    for restart in range(max_restarts):
        shuffled = customers[:]
        random.shuffle(shuffled)
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0 for _ in range(truck_count)]
        unassigned = set(shuffled)
        regret_level = 3 if len(unassigned) > 0.5 * n else 2
        
        # Initial construction with composite regret (max route distance + small weight on route length)
        while unassigned:
            best_customer = None
            best_route_idx = -1
            best_pos = -1
            best_regret = -1.0
            best_composite = float('inf')
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
                        composite = new_max + 0.01 * route_lengths[r_idx]  # encourage shorter routes
                        if composite < best_local_cost:
                            best_local_cost = composite
                            best_local_pos = pos
                    insertions.append((best_local_cost, best_local_pos, r_idx))
                insertions.sort(key=lambda x: (x[0], route_lengths[x[2]], x[1]))
                if not insertions:
                    continue
                best_cost = insertions[0][0]
                second_best_cost = insertions[1][0] if len(insertions) > 1 else best_cost
                third_best_cost = insertions[2][0] if len(insertions) > 2 else best_cost
                regret = second_best_cost - best_cost
                if len(insertions) > 2:
                    regret += (third_best_cost - best_cost) / 2.0
                if regret > best_regret or (regret == best_regret and best_cost < best_composite):
                    best_regret = regret
                    best_customer = c
                    best_composite = best_cost
                    best_route_idx = insertions[0][2]
                    best_pos = insertions[0][1]
            
            if best_customer is None:
                best_customer = next(iter(unassigned))
                best_composite = float('inf')
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        increase = distance_matrix[prev, best_customer] + distance_matrix[best_customer, nxt] - distance_matrix[prev, nxt]
                        new_len = route_lengths[r_idx] + increase
                        new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                        composite = new_max + 0.01 * route_lengths[r_idx]
                        if composite < best_composite:
                            best_composite = composite
                            best_route_idx = r_idx
                            best_pos = pos
            
            route = routes[best_route_idx]
            route.insert(best_pos, best_customer)
            route_lengths[best_route_idx] = compute_route_length(route)
            unassigned.remove(best_customer)
            report_best_vrp(routes)
            regret_level = 3 if len(unassigned) > 0.5 * n else 2
        
        # Local search: intra-route 2-opt, inter-route relocate and exchange, focusing on max route
        max_iterations = 5 * n * truck_count + 10
        improved = True
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            # Intra-route 2-opt
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
            # Inter-route relocate: move a customer from longest route to another if reduces max
            max_len = max(route_lengths)
            longest_indices = [i for i, l in enumerate(route_lengths) if l == max_len]
            for r_from in longest_indices:
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
                            if new_max < max_len - 1e-9:
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
            # Inter-route exchange: swap customers between routes if reduces max
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
        
        # Update best solution
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [route[:] for route in routes]
        
        # Shake: destroy a percentage of customers from the longest routes and reinsert with composite regret
        shake_iterations = max(1, int(0.2 * n / truck_count))
        for _ in range(shake_iterations):
            # Determine percentage (15-25%) and number to remove
            pct = random.uniform(0.15, 0.25)
            num_to_remove = max(1, int(pct * (n - 1)))
            # Preferentially remove from longest routes
            sorted_route_indices = sorted(range(truck_count), key=lambda i: -route_lengths[i])
            removed = []
            remaining = num_to_remove
            for r_idx in sorted_route_indices:
                route = routes[r_idx]
                if len(route) <= 2:
                    continue
                max_from_route = min(remaining, len(route)-2)
                if max_from_route <= 0:
                    break
                # Select customers to remove from this route (random)
                candidates = [c for c in route[1:-1]]
                if len(candidates) > max_from_route:
                    remove_from_route = random.sample(candidates, max_from_route)
                else:
                    remove_from_route = candidates[:]
                removed.extend(remove_from_route)
                remaining -= len(remove_from_route)
                if remaining <= 0:
                    break
            if not removed:
                continue
            # Remove them from routes
            for c in removed:
                for r_idx, route in enumerate(routes):
                    if c in route:
                        idx = route.index(c)
                        prev = route[idx-1]
                        nxt = route[idx+1]
                        cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                        route.pop(idx)
                        route_lengths[r_idx] -= cost_remove
                        break
            # Reinsert using composite regret
            unassigned = set(removed)
            regret_level = 3 if len(unassigned) > 0.5 * n else 2
            while unassigned:
                best_customer = None
                best_route_idx = -1
                best_pos = -1
                best_regret = -1.0
                best_composite = float('inf')
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
                            composite = new_max + 0.01 * route_lengths[r_idx]
                            if composite < best_local_cost:
                                best_local_cost = composite
                                best_local_pos = pos
                        insertions.append((best_local_cost, best_local_pos, r_idx))
                    insertions.sort(key=lambda x: (x[0], route_lengths[x[2]], x[1]))
                    if not insertions:
                        continue
                    best_cost = insertions[0][0]
                    second_best_cost = insertions[1][0] if len(insertions) > 1 else best_cost
                    third_best_cost = insertions[2][0] if len(insertions) > 2 else best_cost
                    regret = second_best_cost - best_cost
                    if len(insertions) > 2:
                        regret += (third_best_cost - best_cost) / 2.0
                    if regret > best_regret or (regret == best_regret and best_cost < best_composite):
                        best_regret = regret
                        best_customer = c
                        best_composite = best_cost
                        best_route_idx = insertions[0][2]
                        best_pos = insertions[0][1]
                
                if best_customer is None:
                    best_customer = next(iter(unassigned))
                    best_composite = float('inf')
                    for r_idx, route in enumerate(routes):
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            increase = distance_matrix[prev, best_customer] + distance_matrix[best_customer, nxt] - distance_matrix[prev, nxt]
                            new_len = route_lengths[r_idx] + increase
                            new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                            composite = new_max + 0.01 * route_lengths[r_idx]
                            if composite < best_composite:
                                best_composite = composite
                                best_route_idx = r_idx
                                best_pos = pos
                
                route = routes[best_route_idx]
                route.insert(best_pos, best_customer)
                route_lengths[best_route_idx] = compute_route_length(route)
                unassigned.remove(best_customer)
                report_best_vrp(routes)
                regret_level = 3 if len(unassigned) > 0.5 * n else 2
            
            # Local search after reinsertion
            max_iterations2 = 3 * n * truck_count + 10
            improved = True
            while improved and max_iterations2 > 0:
                improved = False
                max_iterations2 -= 1
                # Intra-route 2-opt
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
                # Inter-route relocate (prioritize longest routes)
                max_len = max(route_lengths)
                longest_indices = [i for i, l in enumerate(route_lengths) if l == max_len]
                for r_from in longest_indices:
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
                                if new_max < max_len - 1e-9:
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
                # Inter-route exchange
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
            # Update best solution after shake
            current_max = max(route_lengths)
            if current_max < best_max:
                best_max = current_max
                best_routes = [route[:] for route in routes]
    
    if best_routes is None:
        best_routes = routes
    return best_routes