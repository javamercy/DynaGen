import numpy as np
import random
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    
    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    global_best_max = float('inf')
    global_best_routes = None
    num_restarts = 20
    no_improve_restarts = 0
    
    for restart in range(num_restarts):
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        current_dist = [0.0 for _ in range(truck_count)]
        
        # Adaptive lambda based on current imbalance
        def compute_lambda():
            maxd = max(current_dist) if current_dist else 0
            mind = min(current_dist) if current_dist else 0
            if mind < 1e-9:
                return 1.0
            imbalance = (maxd - mind) / mind
            return 1.0 + imbalance
        
        # Regret-2 construction with adaptive balancing
        for _ in range(len(customers)):
            best_regret = -float('inf')
            best_cust = None
            best_route = -1
            best_pos = -1
            lam = compute_lambda()
            for cust in customers:
                costs = []
                for r in range(truck_count):
                    route = routes[r]
                    for i in range(1, len(route)):
                        cost_inc = (distance_matrix[route[i-1], cust] + distance_matrix[cust, route[i]] - distance_matrix[route[i-1], route[i]])
                        new_dist = current_dist[r] + cost_inc
                        other_dists = [current_dist[rr] for rr in range(truck_count) if rr != r]
                        new_max = max(new_dist, max(other_dists)) if other_dists else new_dist
                        cur_max = max(current_dist)
                        penalty = new_max - cur_max
                        total_cost = cost_inc + lam * penalty
                        costs.append((total_cost, i, r, cost_inc))
                if not costs:
                    continue
                costs.sort(key=lambda x: x[0])
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = costs[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_route = costs[0][2]
                    best_pos = costs[0][1]
            if best_cust is None:
                break
            routes[best_route].insert(best_pos, best_cust)
            current_dist[best_route] = route_distance(routes[best_route])
            customers.remove(best_cust)
        
        best_max = max(current_dist)
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [list(r) for r in routes]
            report_best_vrp(routes)
            no_improve_restarts = 0
        else:
            no_improve_restarts += 1
        
        # Local search with non-worsening acceptance
        n_cust = n - 1
        max_iters = 5 * n_cust * truck_count
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Relocate
            routes_new = [list(r) for r in routes]
            dist_new = list(current_dist)
            for r1 in range(truck_count):
                route1 = routes_new[r1]
                if len(route1) <= 2:
                    continue
                for idx in range(1, len(route1)-1):
                    cust = route1[idx]
                    new_route1 = route1[:idx] + route1[idx+1:]
                    new_dist1 = route_distance(new_route1)
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes_new[r2]
                        best_total = float('inf')
                        best_pos2 = -1
                        for i in range(1, len(route2)):
                            cost_inc = (distance_matrix[route2[i-1], cust] + distance_matrix[cust, route2[i]] - distance_matrix[route2[i-1], route2[i]])
                            new_dist2 = dist_new[r2] + cost_inc
                            other_dists = [dist_new[rr] for rr in range(truck_count) if rr not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists) if other_dists else max(new_dist1, new_dist2)
                            if new_max < best_total:
                                best_total = new_max
                                best_pos2 = i
                        if best_total <= best_max:  # non-worsening
                            # accept
                            routes_new[r1] = new_route1
                            routes_new[r2] = route2[:best_pos2] + [cust] + route2[best_pos2:]
                            dist_new[r1] = new_dist1
                            dist_new[r2] = route_distance(routes_new[r2])
                            if best_total < best_max:
                                best_max = best_total
                                improved = True
                                if best_total < global_best_max:
                                    global_best_max = best_total
                                    global_best_routes = [list(r) for r in routes_new]
                                    report_best_vrp(routes_new)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                routes = routes_new
                current_dist = dist_new
                continue
            
            # Swap
            for r1 in range(truck_count):
                route1 = routes_new[r1]
                if len(route1) <= 2:
                    continue
                for idx1 in range(1, len(route1)-1):
                    cust1 = route1[idx1]
                    for r2 in range(r1+1, truck_count):
                        route2 = routes_new[r2]
                        if len(route2) <= 2:
                            continue
                        for idx2 in range(1, len(route2)-1):
                            cust2 = route2[idx2]
                            new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                            new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                            new_dist1 = route_distance(new_route1)
                            new_dist2 = route_distance(new_route2)
                            other_dists = [dist_new[rr] for rr in range(truck_count) if rr not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists) if other_dists else max(new_dist1, new_dist2)
                            if new_max <= best_max:
                                routes_new[r1] = new_route1
                                routes_new[r2] = new_route2
                                dist_new[r1] = new_dist1
                                dist_new[r2] = new_dist2
                                if new_max < best_max:
                                    best_max = new_max
                                    improved = True
                                    if new_max < global_best_max:
                                        global_best_max = new_max
                                        global_best_routes = [list(r) for r in routes_new]
                                        report_best_vrp(routes_new)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                routes = routes_new
                current_dist = dist_new
                continue
            
            # Intra-route 2-opt
            for r in range(truck_count):
                route = routes_new[r]
                if len(route) <= 3:
                    continue
                best_improve = 0
                best_i = best_j = -1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < current_dist[r] - 1e-9:
                            improvement = current_dist[r] - new_dist
                            if improvement > best_improve:
                                best_improve = improvement
                                best_i, best_j = i, j
                if best_improve > 0:
                    new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                    routes_new[r] = new_route
                    dist_new[r] = route_distance(new_route)
                    new_max = max(dist_new)
                    if new_max <= best_max:
                        if new_max < best_max:
                            best_max = new_max
                            improved = True
                            if new_max < global_best_max:
                                global_best_max = new_max
                                global_best_routes = [list(r) for r in routes_new]
                                report_best_vrp(routes_new)
                        else:
                            # non-worsening acceptance
                            improved = True
                    break
            if improved:
                routes = routes_new
                current_dist = dist_new
                continue
            
            # Cross-route 2-opt
            for r1 in range(truck_count):
                route1 = routes_new[r1]
                if len(route1) <= 2:
                    continue
                for r2 in range(r1+1, truck_count):
                    route2 = routes_new[r2]
                    if len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i+1] + route2[j+1:]
                            new2 = route2[:j+1] + route1[i+1:]
                            new_dist1 = route_distance(new1)
                            new_dist2 = route_distance(new2)
                            other_dists = [dist_new[rr] for rr in range(truck_count) if rr not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists) if other_dists else max(new_dist1, new_dist2)
                            if new_max <= best_max:
                                routes_new[r1] = new1
                                routes_new[r2] = new2
                                dist_new[r1] = new_dist1
                                dist_new[r2] = new_dist2
                                if new_max < best_max:
                                    best_max = new_max
                                    improved = True
                                    if new_max < global_best_max:
                                        global_best_max = new_max
                                        global_best_routes = [list(r) for r in routes_new]
                                        report_best_vrp(routes_new)
                                else:
                                    improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                routes = routes_new
                current_dist = dist_new
                continue
        
        # Post-optimization balancing: move from longest route to shorter if reduces max
        for _ in range(n_cust):
            max_idx = max(range(truck_count), key=lambda i: current_dist[i])
            min_idx = min(range(truck_count), key=lambda i: current_dist[i])
            if current_dist[max_idx] - current_dist[min_idx] < 1e-9:
                break
            route_max = routes[max_idx]
            if len(route_max) <= 2:
                break
            found = False
            for idx in range(1, len(route_max)-1):
                cust = route_max[idx]
                new_route_max = route_max[:idx] + route_max[idx+1:]
                new_dist_max = route_distance(new_route_max)
                route_min = routes[min_idx]
                best_pos_min = -1
                best_new_dist_min = float('inf')
                for i in range(1, len(route_min)):
                    cost_inc = distance_matrix[route_min[i-1], cust] + distance_matrix[cust, route_min[i]] - distance_matrix[route_min[i-1], route_min[i]]
                    new_dist_min = current_dist[min_idx] + cost_inc
                    if new_dist_min < best_new_dist_min:
                        best_new_dist_min = new_dist_min
                        best_pos_min = i
                other_dists = [current_dist[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max = max(new_dist_max, best_new_dist_min, *other_dists)
                if new_max < best_max:
                    routes[max_idx] = new_route_max
                    routes[min_idx] = route_min[:best_pos_min] + [cust] + route_min[best_pos_min:]
                    current_dist[max_idx] = new_dist_max
                    current_dist[min_idx] = best_new_dist_min
                    best_max = new_max
                    if best_max < global_best_max:
                        global_best_max = best_max
                        global_best_routes = [list(r) for r in routes]
                        report_best_vrp(routes)
                    found = True
                    break
            if not found:
                break
        
        # Diversity trigger: if no improvement for 5 restarts, shuffle more aggressively (already done each restart)
        if no_improve_restarts >= 5:
            # Increase randomness: reinitialize routes randomly? Not needed, shuffle already.
            pass
    return global_best_routes