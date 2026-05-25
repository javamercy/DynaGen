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
    
    def compute_lambda(current_dist):
        # adaptive lambda based on imbalance
        max_dist = max(current_dist)
        min_dist = min(current_dist)
        if max_dist > 0:
            imbalance = (max_dist - min_dist) / max_dist
        else:
            imbalance = 0.0
        # base_lambda = 1.0, scale factor = 0.5
        return 1.0 + 0.5 * imbalance
    
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
            # Compute lambda based on current distances one time per insertion
            lam = compute_lambda(current_dist)
            best_max_current = max(current_dist)
            for cust in customers:
                costs = []
                for r in range(truck_count):
                    route = routes[r]
                    for i in range(1, len(route)):
                        cost_inc = distance_matrix[route[i-1], cust] + distance_matrix[cust, route[i]] - distance_matrix[route[i-1], route[i]]
                        new_dist = current_dist[r] + cost_inc
                        new_max = max(new_dist, max(current_dist[:r] + current_dist[r+1:]))
                        # balancing penalty: penalize increase in max
                        penalty = new_max - best_max_current
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
        
        # Local search with non-worsening acceptance
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
                            # Non-worsening acceptance: accept if new_max <= best_max
                            if new_max < best_total:
                                best_total = new_max
                                best_pos2 = i
                        if best_total <= best_max:  # non-worsening
                            if best_total < best_max:
                                routes[r1] = new_route1
                                routes[r2] = route2[:best_pos2] + [cust] + route2[best_pos2:]
                                current_dist[r1] = new_dist1
                                current_dist[r2] = route_distance(routes[r2])
                                best_max = best_total
                                if best_max < global_best_max:
                                    global_best_max = best_max
                                    global_best_routes = [list(r) for r in routes]
                                    report_best_vrp(routes)
                                improved = True
                            # If equal, we still accept to diversify (non-worsening)
                            # But careful: accept only if it doesn't cause infinite loop? We'll allow strictly <= but only if improvement flag set? For equality, we'll not set improved to avoid cycling, but we can still apply move.
                            # Actually we want to accept non-worsening moves even if equal, but not set improved (to avoid infinite loop). So we need a separate flag for improvement.
                            # Let's handle: if best_total < best_max: improved=True; else if best_total == best_max: apply move but don't set improved.
                            if best_total < best_max:
                                improved = True
                            else:
                                # Apply move anyway
                                pass
                            break
                    if best_total <= best_max:
                        break
                if best_total <= best_max:
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
                                    if best_max < global_best_max:
                                        global_best_max = best_max
                                        global_best_routes = [list(r) for r in routes]
                                        report_best_vrp(routes)
                                # If equal, apply but no improvement flag
                                break
                    if new_max <= best_max:
                        break
                if new_max <= best_max:
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
                            if best_max < global_best_max:
                                global_best_max = best_max
                                global_best_routes = [list(r) for r in routes]
                                report_best_vrp(routes)
                        # If equal, apply move but no improvement
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
                                    if best_max < global_best_max:
                                        global_best_max = best_max
                                        global_best_routes = [list(r) for r in routes]
                                        report_best_vrp(routes)
                                # If equal, apply but no improvement
                                break
                    if new_max <= best_max:
                        break
                if new_max <= best_max:
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