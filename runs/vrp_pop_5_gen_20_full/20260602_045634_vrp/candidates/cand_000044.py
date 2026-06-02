import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_distance(routes):
        return max(route_distance(r) for r in routes)
    
    global_best_routes = None
    global_best_max = float('inf')
    
    restarts = max(3, n // 10)
    for restart in range(restarts):
        # Regret-2 construction
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        random.shuffle(unassigned)  # initial random order to break ties
        while unassigned:
            best_cust = None
            best_regret = -float('inf')
            best_info = None  # (route_idx, pos, inc)
            for cust in unassigned:
                # Compute insertion costs for all possible positions
                insertions = []
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        added = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        # Compute new max distance if inserted here
                        current_max = max_distance(routes)
                        new_route_dist = route_distance(route) + added
                        other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                        new_max = max(new_route_dist, other_max)
                        increase = new_max - current_max
                        insertions.append((increase, r_idx, pos))
                # Regret-2: difference between best and second best insertion increase
                insertions.sort()
                if len(insertions) == 1:
                    regret = 0
                else:
                    regret = insertions[1][0] - insertions[0][0]
                # Choose customer with highest regret (tie break by smaller increase)
                if regret > best_regret or (regret == best_regret and insertions[0][0] < best_info[0]):
                    best_regret = regret
                    best_cust = cust
                    best_info = insertions[0]
            # Insert best customer
            r_idx, pos = best_info[1], best_info[2]
            routes[r_idx].insert(pos, best_cust)
            unassigned.remove(best_cust)
        
        best_routes = [r[:] for r in routes]
        best_max = max_distance(routes)
        report_best_vrp(best_routes)
        
        # Variable Neighborhood Descent (VND) with neighborhoods: 2-opt, relocate, swap, cross
        max_iter = (n - 1) * truck_count * 5
        no_improve = 0
        for _ in range(max_iter):
            improved = False
            # Apply each neighborhood in order; if improvement, restart from first
            for phase in ['2opt', 'relocate', 'swap', 'cross']:
                if phase == '2opt':
                    for r_idx in range(truck_count):
                        route = routes[r_idx]
                        if len(route) <= 3:
                            continue
                        for i in range(1, len(route)-2):
                            for j in range(i+1, len(route)-1):
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                old_dist = route_distance(route)
                                new_dist = route_distance(new_route)
                                if new_dist >= old_dist:
                                    continue
                                other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != r_idx)
                                new_max = max(new_dist, other_max)
                                if new_max < best_max:
                                    routes[r_idx] = new_route
                                    best_routes = [r[:] for r in routes]
                                    best_max = new_max
                                    improved = True
                                    report_best_vrp(best_routes)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == 'relocate':
                    for src in range(truck_count):
                        route_src = routes[src]
                        if len(route_src) <= 2:
                            continue
                        for pos_src in range(1, len(route_src)-1):
                            cust = route_src[pos_src]
                            temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                            dist_src = route_distance(temp_src)
                            for dst in range(truck_count):
                                if dst == src:
                                    continue
                                route_dst = routes[dst]
                                for pos_dst in range(1, len(route_dst)):
                                    new_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                                    dist_dst = route_distance(new_dst)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != src and x != dst)
                                    new_max = max(dist_src, dist_dst, other_max)
                                    if new_max < best_max:
                                        routes[src] = temp_src
                                        routes[dst] = new_dst
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == 'swap':
                    for t1 in range(truck_count):
                        route1 = routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
                            route2 = routes[t2]
                            if len(route2) <= 2:
                                continue
                            for i in range(1, len(route1)-1):
                                for j in range(1, len(route2)-1):
                                    cust1 = route1[i]
                                    cust2 = route2[j]
                                    new_route1 = route1[:i] + [cust2] + route1[i+1:]
                                    new_route2 = route2[:j] + [cust1] + route2[j+1:]
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                elif phase == 'cross':
                    for t1 in range(truck_count):
                        route1 = routes[t1]
                        if len(route1) <= 2:
                            continue
                        for t2 in range(t1+1, truck_count):
                            route2 = routes[t2]
                            if len(route2) <= 2:
                                continue
                            for i in range(1, len(route1)-1):
                                for j in range(1, len(route2)-1):
                                    new_route1 = route1[:i] + route2[j:]
                                    new_route2 = route2[:j] + route1[i:]
                                    dist1 = route_distance(new_route1)
                                    dist2 = route_distance(new_route2)
                                    other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                                    new_max = max(dist1, dist2, other_max)
                                    if new_max < best_max:
                                        routes[t1] = new_route1
                                        routes[t2] = new_route2
                                        best_routes = [r[:] for r in routes]
                                        best_max = new_max
                                        improved = True
                                        report_best_vrp(best_routes)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    break
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 3:
                    break
        
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [r[:] for r in best_routes]
            report_best_vrp(global_best_routes)
    
    if global_best_routes is None:
        # Fallback to simple construction if somehow no solution found (shouldn't happen)
        routes = [[0, i, 0] for i in range(1, min(truck_count+1, n))]
        for i in range(truck_count - (n-1)):
            routes.append([0,0])
        global_best_routes = routes
        report_best_vrp(global_best_routes)
    return global_best_routes