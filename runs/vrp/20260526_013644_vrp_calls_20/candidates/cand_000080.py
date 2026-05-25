import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def compute_max(routes):
        return max(route_distance(r) for r in routes)
    
    def construct(reverse_ties=False):
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0 for _ in range(truck_count)]
        unassigned = set(customers)
        while unassigned:
            best_cust = None
            best_regret = -float('inf')
            best_tie_breaker = float('inf')
            best_route_idx = None
            best_pos = None
            best_cost = None
            for cust in list(unassigned):
                costs = []
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        delta = (distance_matrix[route[pos-1], cust] +
                                 distance_matrix[cust, route[pos]] -
                                 distance_matrix[route[pos-1], route[pos]])
                        costs.append((route_dists[r] + delta, r, pos))
                costs.sort(key=lambda x: x[0])
                best = costs[0]
                second = costs[1] if len(costs) > 1 else (float('inf'), -1, -1)
                regret = second[0] - best[0]
                # tie-break by customer id
                tie_breaker = cust
                if reverse_ties:
                    tie_breaker = -cust
                if (regret > best_regret or
                    (regret == best_regret and tie_breaker < best_tie_breaker)):
                    best_regret = regret
                    best_tie_breaker = tie_breaker
                    best_cust = cust
                    best_route_idx = best[1]
                    best_pos = best[2]
                    best_cost = best[0]
            routes[best_route_idx].insert(best_pos, best_cust)
            route_dists[best_route_idx] = best_cost
            unassigned.remove(best_cust)
        return routes, route_dists
    
    # Initial solution
    routes, route_dists = construct(reverse_ties=False)
    best_routes = [r[:] for r in routes]
    best_max = compute_max(routes)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass
    
    current_routes = [r[:] for r in routes]
    current_max = best_max
    deviation = 0.1 * current_max
    max_iter = 600
    no_improve = 0
    
    for it in range(max_iter):
        improved = False
        # Inter-route relocate
        best_move = None
        best_new_max = current_max + deviation + 1
        for r_from in range(truck_count):
            if len(current_routes[r_from]) <= 2:
                continue
            for pos_from in range(1, len(current_routes[r_from]) - 1):
                cust = current_routes[r_from][pos_from]
                prev = current_routes[r_from][pos_from - 1]
                nxt = current_routes[r_from][pos_from + 1]
                delta_from = (distance_matrix[prev, nxt] -
                               distance_matrix[prev, cust] -
                               distance_matrix[cust, nxt])
                new_from_dist = route_dists[r_from] + delta_from
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    route_to = current_routes[r_to]
                    for pos_to in range(1, len(route_to)):
                        prev_to = route_to[pos_to - 1]
                        nxt_to = route_to[pos_to]
                        delta_to = (distance_matrix[prev_to, cust] +
                                     distance_matrix[cust, nxt_to] -
                                     distance_matrix[prev_to, nxt_to])
                        new_to_dist = route_dists[r_to] + delta_to
                        cand = max(new_from_dist, new_to_dist)
                        for rr in range(truck_count):
                            if rr not in (r_from, r_to):
                                cand = max(cand, route_dists[rr])
                        if cand < best_new_max:
                            best_new_max = cand
                            best_move = (r_from, pos_from, r_to, pos_to,
                                         new_from_dist, new_to_dist)
        if best_move and best_new_max <= current_max + deviation:
            r_f, p_f, r_t, p_t, d_f, d_t = best_move
            cust = current_routes[r_f].pop(p_f)
            current_routes[r_t].insert(p_t, cust)
            route_dists[r_f] = d_f
            route_dists[r_t] = d_t
            current_max = compute_max(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            improved = True
        
        # Inter-route swap
        best_move = None
        best_new_max = current_max + deviation + 1
        for r1 in range(truck_count):
            if len(current_routes[r1]) <= 2:
                continue
            for pos1 in range(1, len(current_routes[r1]) - 1):
                cust1 = current_routes[r1][pos1]
                prev1 = current_routes[r1][pos1 - 1]
                nxt1 = current_routes[r1][pos1 + 1]
                for r2 in range(r1 + 1, truck_count):
                    if len(current_routes[r2]) <= 2:
                        continue
                    for pos2 in range(1, len(current_routes[r2]) - 1):
                        cust2 = current_routes[r2][pos2]
                        prev2 = current_routes[r2][pos2 - 1]
                        nxt2 = current_routes[r2][pos2 + 1]
                        delta1 = (distance_matrix[prev1, cust2] +
                                   distance_matrix[cust2, nxt1] -
                                   distance_matrix[prev1, cust1] -
                                   distance_matrix[cust1, nxt1])
                        delta2 = (distance_matrix[prev2, cust1] +
                                   distance_matrix[cust1, nxt2] -
                                   distance_matrix[prev2, cust2] -
                                   distance_matrix[cust2, nxt2])
                        new_dist1 = route_dists[r1] + delta1
                        new_dist2 = route_dists[r2] + delta2
                        cand = max(new_dist1, new_dist2)
                        for rr in range(truck_count):
                            if rr not in (r1, r2):
                                cand = max(cand, route_dists[rr])
                        if cand < best_new_max:
                            best_new_max = cand
                            best_move = (r1, pos1, r2, pos2,
                                         new_dist1, new_dist2, cust1, cust2)
        if best_move and best_new_max <= current_max + deviation:
            r1, p1, r2, p2, d1, d2, c1, c2 = best_move
            current_routes[r1][p1] = c2
            current_routes[r2][p2] = c1
            route_dists[r1] = d1
            route_dists[r2] = d2
            current_max = compute_max(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            improved = True
        
        if not improved:
            no_improve += 1
            deviation *= 0.98
        else:
            no_improve = 0
        
        # Restart after 100 iterations without improvement
        if no_improve >= 100:
            routes2, dists2 = construct(reverse_ties=True)
            current_routes = [r[:] for r in routes2]
            route_dists = dists2[:]
            current_max = compute_max(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            deviation = 0.1 * current_max
            no_improve = 0
    
    return best_routes