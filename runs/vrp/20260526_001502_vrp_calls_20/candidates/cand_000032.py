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
    
    def insert_cost(route, node):
        best_cost = float('inf')
        best_pos = -1
        for i in range(1, len(route)):
            cost = distance_matrix[route[i-1], node] + distance_matrix[node, route[i]] - distance_matrix[route[i-1], route[i]]
            if cost < best_cost:
                best_cost = cost
                best_pos = i
        return best_cost, best_pos
    
    global_best_max = float('inf')
    global_best_routes = None
    num_restarts = 10
    balance_alpha = 1.0
    
    for restart in range(num_restarts):
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        current_dist = [0.0 for _ in range(truck_count)]
        
        # Regret-2 insertion with balancing penalty
        for _ in range(len(customers)):
            best_regret = -float('inf')
            best_cust = None
            best_route = -1
            best_pos = -1
            current_max = max(current_dist) if current_dist else 0.0
            for cust in customers:
                costs = []
                for r in range(truck_count):
                    cost, pos = insert_cost(routes[r], cust)
                    new_dist_r = current_dist[r] + cost
                    delta = max(new_dist_r, current_max) - current_max
                    penalized_cost = cost + balance_alpha * delta
                    costs.append((penalized_cost, cost, pos, r))
                costs.sort(key=lambda x: x[0])
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = costs[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_route = costs[0][3]
                    best_pos = costs[0][2]
            # Insert best_cust
            routes[best_route].insert(best_pos, best_cust)
            current_dist[best_route] = route_distance(routes[best_route])
            customers.remove(best_cust)
        
        best_max = max(current_dist)
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [list(r) for r in routes]
            report_best_vrp(routes)
        
        # Local search (same as parent)
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
                        cost, pos = insert_cost(route2, cust)
                        new_dist2 = current_dist[r2] + cost
                        other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1, r2)]
                        new_max = max(new_dist1, new_dist2, *other_dists)
                        if new_max < best_max:
                            routes[r1] = new_route1
                            routes[r2] = route2[:pos] + [cust] + route2[pos:]
                            current_dist[r1] = new_dist1
                            current_dist[r2] = new_dist2
                            best_max = new_max
                            if new_max < global_best_max:
                                global_best_max = new_max
                                global_best_routes = [list(r) for r in routes]
                                report_best_vrp(routes)
                            improved = True
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
                            if new_max < best_max:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                best_max = new_max
                                if new_max < global_best_max:
                                    global_best_max = new_max
                                    global_best_routes = [list(r) for r in routes]
                                    report_best_vrp(routes)
                                improved = True
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
                    if new_max < best_max:
                        best_max = new_max
                        if new_max < global_best_max:
                            global_best_max = new_max
                            global_best_routes = [list(r) for r in routes]
                            report_best_vrp(routes)
                    improved = True
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
                            if new_max < best_max:
                                routes[r1] = new1
                                routes[r2] = new2
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                best_max = new_max
                                if new_max < global_best_max:
                                    global_best_max = new_max
                                    global_best_routes = [list(r) for r in routes]
                                    report_best_vrp(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Post-optimization balancing step
        max_bal_iters = 5 * n_cust
        for _ in range(max_bal_iters):
            max_idx = max(range(truck_count), key=lambda r: current_dist[r])
            min_idx = min(range(truck_count), key=lambda r: current_dist[r])
            if max_idx == min_idx:
                break
            route_max = routes[max_idx]
            route_min = routes[min_idx]
            best_improvement = 0
            best_move = None
            # Try relocate from max to min
            for idx in range(1, len(route_max)-1):
                cust = route_max[idx]
                new_route_max = route_max[:idx] + route_max[idx+1:]
                new_dist_max = route_distance(new_route_max)
                cost, pos = insert_cost(route_min, cust)
                new_dist_min = current_dist[min_idx] + cost
                new_max = max(new_dist_max, new_dist_min, *[current_dist[i] for i in range(truck_count) if i not in (max_idx, min_idx)])
                if new_max < best_max:
                    improvement = best_max - new_max
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = ('relocate', idx, cust, pos, new_route_max, new_dist_max, new_dist_min)
            # Try swap between max and min
            for idx1 in range(1, len(route_max)-1):
                cust1 = route_max[idx1]
                for idx2 in range(1, len(route_min)-1):
                    cust2 = route_min[idx2]
                    new_route_max = route_max[:idx1] + [cust2] + route_max[idx1+1:]
                    new_route_min = route_min[:idx2] + [cust1] + route_min[idx2+1:]
                    new_dist_max = route_distance(new_route_max)
                    new_dist_min = route_distance(new_route_min)
                    new_max = max(new_dist_max, new_dist_min, *[current_dist[i] for i in range(truck_count) if i not in (max_idx, min_idx)])
                    if new_max < best_max:
                        improvement = best_max - new_max
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = ('swap', idx1, idx2, cust1, cust2, new_route_max, new_route_min, new_dist_max, new_dist_min)
            if best_move is None:
                break
            if best_move[0] == 'relocate':
                _, idx, cust, pos, new_route_max, new_dist_max, new_dist_min = best_move
                routes[max_idx] = new_route_max
                routes[min_idx] = route_min[:pos] + [cust] + route_min[pos:]
                current_dist[max_idx] = new_dist_max
                current_dist[min_idx] = new_dist_min
                best_max = max(current_dist)
                if best_max < global_best_max:
                    global_best_max = best_max
                    global_best_routes = [list(r) for r in routes]
                    report_best_vrp(routes)
            else: # swap
                _, idx1, idx2, cust1, cust2, new_route_max, new_route_min, new_dist_max, new_dist_min = best_move
                routes[max_idx] = new_route_max
                routes[min_idx] = new_route_min
                current_dist[max_idx] = new_dist_max
                current_dist[min_idx] = new_dist_min
                best_max = max(current_dist)
                if best_max < global_best_max:
                    global_best_max = best_max
                    global_best_routes = [list(r) for r in routes]
                    report_best_vrp(routes)
        
        # Final update if balancing improved
        if max(current_dist) < global_best_max:
            global_best_max = max(current_dist)
            global_best_routes = [list(r) for r in routes]
            report_best_vrp(routes)
    
    return global_best_routes