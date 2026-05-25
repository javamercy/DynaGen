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
    
    def insert_cost_balanced(routes, node, current_dists, best_max):
        max_dist = max(current_dists)
        min_dist = min(current_dists)
        if max_dist > 0:
            imbalance = (max_dist - min_dist) / max_dist
        else:
            imbalance = 0.0
        lambda_factor = 1.0 + imbalance
        best_total_cost = float('inf')
        best_route = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            for i in range(1, len(route)):
                cost_inc = distance_matrix[route[i-1], node] + distance_matrix[node, route[i]] - distance_matrix[route[i-1], route[i]]
                new_dist = current_dists[r] + cost_inc
                new_max = max(new_dist, max(current_dists[:r] + current_dists[r+1:]))
                penalty = lambda_factor * (new_max - best_max)
                total_cost = cost_inc + penalty
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_route = r
                    best_pos = i
        return best_total_cost, best_route, best_pos
    
    global_best_max = float('inf')
    global_best_routes = None
    num_restarts = 20
    
    for restart in range(num_restarts):
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        current_dist = [0.0 for _ in range(truck_count)]
        
        # Regret-2 insertion with adaptive balancing penalty
        for _ in range(len(customers)):
            best_regret = -float('inf')
            best_cust = None
            best_route = -1
            best_pos = -1
            for cust in customers:
                costs = []
                for r in range(truck_count):
                    route = routes[r]
                    for i in range(1, len(route)):
                        cost_inc = distance_matrix[route[i-1], cust] + distance_matrix[cust, route[i]] - distance_matrix[route[i-1], route[i]]
                        new_dist = current_dist[r] + cost_inc
                        new_max = max(new_dist, max(current_dist[:r] + current_dist[r+1:]))
                        penalty = (1.0 + (max(current_dist) - min(current_dist)) / max(current_dist) if max(current_dist) > 0 else 0.0) * (new_max - max(current_dist))
                        total_cost = cost_inc + penalty
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
        
        # Local search
        n_cust = n - 1
        max_iters = 10 * n_cust * truck_count
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Relocate
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for idx in range(1, len(route1)-1):
                    cust = route1[idx]
                    new_route1 = route1[:idx] + route1[idx+1:]
                    new_dist1 = route_distance(new_route1)
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes[r2]
                        best_total = float('inf')
                        best_pos2 = -1
                        for i in range(1, len(route2)):
                            cost_inc = distance_matrix[route2[i-1], cust] + distance_matrix[cust, route2[i]] - distance_matrix[route2[i-1], route2[i]]
                            new_dist2 = current_dist[r2] + cost_inc
                            new_max = max(new_dist1, new_dist2, *[current_dist[i] for i in range(truck_count) if i not in (r1, r2)])
                            total = new_max
                            if total < best_total:
                                best_total = total
                                best_pos2 = i
                        if best_total <= best_max:
                            routes[r1] = new_route1
                            routes[r2] = route2[:best_pos2] + [cust] + route2[best_pos2:]
                            current_dist[r1] = new_dist1
                            current_dist[r2] = route_distance(routes[r2])
                            if best_total < best_max:
                                best_max = best_total
                                improved = True
                                if best_max < global_best_max:
                                    global_best_max = best_max
                                    global_best_routes = [list(r) for r in routes]
                                    report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Swap
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for idx1 in range(1, len(route1)-1):
                    cust1 = route1[idx1]
                    for r2 in range(r1+1, truck_count):
                        route2 = routes[r2]
                        if len(route2) <= 2:
                            continue
                        for idx2 in range(1, len(route2)-1):
                            cust2 = route2[idx2]
                            new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                            new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                            new_dist1 = route_distance(new_route1)
                            new_dist2 = route_distance(new_route2)
                            other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max <= best_max:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                if new_max < best_max:
                                    best_max = new_max
                                    improved = True
                                    if new_max < global_best_max:
                                        global_best_max = new_max
                                        global_best_routes = [list(r) for r in routes]
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
            # Intra-route 2-opt
            for r in range(truck_count):
                route = routes[r]
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
                    routes[r] = new_route
                    current_dist[r] = route_distance(new_route)
                    new_max = max(current_dist)
                    if new_max <= best_max:
                        if new_max < best_max:
                            best_max = new_max
                            improved = True
                            if new_max < global_best_max:
                                global_best_max = new_max
                                global_best_routes = [list(r) for r in routes]
                                report_best_vrp(routes)
                    break
            if improved:
                continue
            # Cross-route 2-opt
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i+1] + route2[j+1:]
                            new2 = route2[:j+1] + route1[i+1:]
                            new_dist1 = route_distance(new1)
                            new_dist2 = route_distance(new2)
                            other_dists = [current_dist[k] for k in range(truck_count) if k not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max <= best_max:
                                routes[r1] = new1
                                routes[r2] = new2
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                if new_max < best_max:
                                    best_max = new_max
                                    improved = True
                                    if new_max < global_best_max:
                                        global_best_max = new_max
                                        global_best_routes = [list(r) for r in routes]
                                        report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        # Post-optimization balancing
        for _ in range(n_cust):
            max_route_idx = max(range(truck_count), key=lambda i: current_dist[i])
            min_route_idx = min(range(truck_count), key=lambda i: current_dist[i])
            if current_dist[max_route_idx] - current_dist[min_route_idx] < 1e-9:
                break
            route_max = routes[max_route_idx]
            if len(route_max) <= 2:
                break
            found = False
            for idx in range(1, len(route_max)-1):
                cust = route_max[idx]
                new_route_max = route_max[:idx] + route_max[idx+1:]
                new_dist_max = route_distance(new_route_max)
                route_min = routes[min_route_idx]
                best_pos_min = -1
                best_new_dist_min = float('inf')
                for i in range(1, len(route_min)):
                    cost_inc = distance_matrix[route_min[i-1], cust] + distance_matrix[cust, route_min[i]] - distance_matrix[route_min[i-1], route_min[i]]
                    new_dist_min = current_dist[min_route_idx] + cost_inc
                    if new_dist_min < best_new_dist_min:
                        best_new_dist_min = new_dist_min
                        best_pos_min = i
                new_max = max(new_dist_max, best_new_dist_min, *[current_dist[i] for i in range(truck_count) if i not in (max_route_idx, min_route_idx)])
                if new_max < best_max:
                    routes[max_route_idx] = new_route_max
                    routes[min_route_idx] = routes[min_route_idx][:best_pos_min] + [cust] + routes[min_route_idx][best_pos_min:]
                    current_dist[max_route_idx] = new_dist_max
                    current_dist[min_route_idx] = best_new_dist_min
                    best_max = new_max
                    if best_max < global_best_max:
                        global_best_max = best_max
                        global_best_routes = [list(r) for r in routes]
                        report_best_vrp(routes)
                    found = True
                    break
            if not found:
                break
    return global_best_routes