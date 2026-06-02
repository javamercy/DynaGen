def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    num_cust = n - 1
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count
    
    # Construction: greedy insertion minimizing max route distance
    for c in customers:
        best_route = -1
        best_pos = -1
        best_max = float('inf')
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                new_len = route_lengths[r_idx] + increase
                new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                if new_max < best_max:
                    best_max = new_max
                    best_route = r_idx
                    best_pos = pos
        route = routes[best_route]
        route.insert(best_pos, c)
        route_lengths[best_route] = sum(distance_matrix[route[k], route[k+1]] for k in range(len(route)-1))
    report_best_vrp(routes)
    
    # Local search: VND with 2-opt, relocate, swap
    max_iter = num_cust * truck_count * 2
    for _ in range(max_iter):
        improved = False
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new < old:
                        new_len = route_lengths[r_idx] - old + new
                        new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                        current_max = max(route_lengths)
                        if new_max < current_max:
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
        
        # Inter-route relocate
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
                    best_pos = -1
                    best_new_len_to = float('inf')
                    for pos in range(1, len(route_to)):
                        prev_to = route_to[pos-1]
                        nxt_to = route_to[pos]
                        cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                        new_len_to = route_lengths[r_to] + cost_insert
                        if new_len_to < best_new_len_to:
                            best_new_len_to = new_len_to
                            best_pos = pos
                    new_max = max(new_len_from, best_new_len_to, [route_lengths[i] for i in range(truck_count) if i not in (r_from, r_to)])
                    if new_max < max(route_lengths):
                        del route_from[idx_c]
                        route_lengths[r_from] = new_len_from
                        route_to.insert(best_pos, c)
                        route_lengths[r_to] = best_new_len_to
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # Inter-route swap
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                c1 = route1[idx1]
                prev1 = route1[idx1-1]
                next1 = route1[idx1+1]
                cost_remove1 = distance_matrix[prev1, c1] + distance_matrix[c1, next1] - distance_matrix[prev1, next1]
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 2:
                        continue
                    for idx2 in range(1, len(route2)-1):
                        c2 = route2[idx2]
                        prev2 = route2[idx2-1]
                        next2 = route2[idx2+1]
                        cost_remove2 = distance_matrix[prev2, c2] + distance_matrix[c2, next2] - distance_matrix[prev2, next2]
                        cost_insert1 = distance_matrix[prev1, c2] + distance_matrix[c2, next1] - distance_matrix[prev1, next1]
                        new_len1 = route_lengths[r1] - cost_remove1 + cost_insert1
                        cost_insert2 = distance_matrix[prev2, c1] + distance_matrix[c1, next2] - distance_matrix[prev2, next2]
                        new_len2 = route_lengths[r2] - cost_remove2 + cost_insert2
                        other_lengths = [route_lengths[i] for i in range(truck_count) if i not in (r1, r2)]
                        new_max = max(new_len1, new_len2, max(other_lengths) if other_lengths else 0)
                        if new_max < max(route_lengths):
                            route1[idx1] = c2
                            route2[idx2] = c1
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
        if not improved:
            break
    
    return routes