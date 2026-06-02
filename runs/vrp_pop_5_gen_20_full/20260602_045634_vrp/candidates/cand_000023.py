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
    restarts = max(5, n // 15)
    
    for restart in range(restarts):
        # Regret-2 construction
        customers = list(range(1, n))
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(customers)
        while unassigned:
            best_regret = -1
            best_cust = None
            best_insertions = None
            for cust in unassigned:
                # compute insertion costs for each route
                costs = []
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    min_cost = float('inf')
                    min_pos = -1
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        if cost < min_cost:
                            min_cost = cost
                            min_pos = pos
                    # compute new route distance if inserted
                    new_route_dist = route_distance(route) + min_cost
                    other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_route_dist, other_max)
                    costs.append((min_cost, min_pos, r_idx, new_max))
                # sort by new_max, pick best two
                costs.sort(key=lambda x: x[3])
                if len(costs) >= 2:
                    regret = costs[1][3] - costs[0][3]
                else:
                    regret = 0
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_insertions = costs[0]
            # insert best_cust
            _, pos, r_idx, _ = best_insertions
            routes[r_idx].insert(pos, best_cust)
            unassigned.remove(best_cust)
        
        best_routes = [r[:] for r in routes]
        best_max = max_distance(routes)
        report_best_vrp(best_routes)
        
        # VND with best improvement
        max_vnd_iters = 5
        for _ in range(max_vnd_iters):
            improved = False
            # 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                best_improve = 0
                best_i = best_j = -1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        old_dist = route_distance(route)
                        new_dist = route_distance(new_route)
                        improve = old_dist - new_dist
                        if improve > best_improve:
                            best_improve = improve
                            best_i, best_j = i, j
                if best_improve > 0:
                    i, j = best_i, best_j
                    routes[r_idx] = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    improved = True
                    new_max = max_distance(routes)
                    if new_max < best_max:
                        best_routes = [r[:] for r in routes]
                        best_max = new_max
                        report_best_vrp(best_routes)
            # relocate
            for src in range(truck_count):
                route_src = routes[src]
                if len(route_src) <= 2:
                    continue
                best_improve = 0
                best_params = None
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
                                improve = best_max - new_max
                                if improve > best_improve:
                                    best_improve = improve
                                    best_params = (src, pos_src, dst, pos_dst, temp_src, new_dst, new_max)
                if best_params:
                    src, pos_src, dst, pos_dst, temp_src, new_dst, new_max = best_params
                    routes[src] = temp_src
                    routes[dst] = new_dst
                    best_max = new_max
                    best_routes = [r[:] for r in routes]
                    improved = True
                    report_best_vrp(best_routes)
            # swap
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for t2 in range(t1+1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    best_improve = 0
                    best_params = None
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
                                improve = best_max - new_max
                                if improve > best_improve:
                                    best_improve = improve
                                    best_params = (t1, t2, i, j, new_route1, new_route2, new_max)
                    if best_params:
                        t1, t2, i, j, new_route1, new_route2, new_max = best_params
                        routes[t1] = new_route1
                        routes[t2] = new_route2
                        best_max = new_max
                        best_routes = [r[:] for r in routes]
                        improved = True
                        report_best_vrp(best_routes)
            # cross
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for t2 in range(t1+1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    best_improve = 0
                    best_params = None
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new_route1 = route1[:i] + route2[j:]
                            new_route2 = route2[:j] + route1[i:]
                            # ensure they start and end at 0? They do because we keep indices
                            dist1 = route_distance(new_route1)
                            dist2 = route_distance(new_route2)
                            other_max = max(route_distance(routes[x]) for x in range(truck_count) if x != t1 and x != t2)
                            new_max = max(dist1, dist2, other_max)
                            if new_max < best_max:
                                improve = best_max - new_max
                                if improve > best_improve:
                                    best_improve = improve
                                    best_params = (t1, t2, i, j, new_route1, new_route2, new_max)
                    if best_params:
                        t1, t2, i, j, new_route1, new_route2, new_max = best_params
                        routes[t1] = new_route1
                        routes[t2] = new_route2
                        best_max = new_max
                        best_routes = [r[:] for r in routes]
                        improved = True
                        report_best_vrp(best_routes)
            if not improved:
                break
        
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [r[:] for r in best_routes]
            report_best_vrp(global_best_routes)
    
    return global_best_routes