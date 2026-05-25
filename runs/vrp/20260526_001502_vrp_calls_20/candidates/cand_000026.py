import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    customers = list(range(1, n))
    best_routes = None
    best_max = float('inf')
    
    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    num_restarts = 5
    for restart in range(num_restarts):
        rng = random.Random(restart)
        cust_order = customers[:]
        rng.shuffle(cust_order)
        # Initialize routes: each truck starts and ends at depot
        routes = [[0, 0] for _ in range(truck_count)]
        remaining = set(customers)
        
        # Regret-2 construction
        while remaining:
            best_regret = -float('inf')
            best_cust = None
            best_insertion = None
            for cust in remaining:
                costs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        cost = (distance_matrix[route[pos-1], cust] +
                                distance_matrix[cust, route[pos]] -
                                distance_matrix[route[pos-1], route[pos]])
                        costs.append((cost, r_idx, pos))
                costs.sort(key=lambda x: x[0])
                best_cost, best_r, best_p = costs[0]
                second_best_cost = costs[1][0] if len(costs) > 1 else best_cost
                regret = second_best_cost - best_cost
                # tie-breaking by order in cust_order
                if regret > best_regret + 1e-9:
                    best_regret = regret
                    best_cust = cust
                    best_insertion = (best_r, best_p)
                elif abs(regret - best_regret) < 1e-9:
                    if cust_order.index(cust) < cust_order.index(best_cust):
                        best_cust = cust
                        best_insertion = (best_r, best_p)
            # Insert best_customer
            r, pos = best_insertion
            routes[r] = routes[r][:pos] + [best_cust] + routes[r][pos:]
            remaining.remove(best_cust)
        
        # Compute current route distances
        current_dist = [route_distance(r) for r in routes]
        local_best_routes = [list(r) for r in routes]
        local_best_max = max(current_dist)
        
        # Local search (first improvement on max distance)
        max_iters = 10 * (n - 1) * truck_count
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
                    old_dist1 = current_dist[r1]
                    new_route1 = route1[:idx] + route1[idx+1:]
                    new_dist1 = route_distance(new_route1)
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes[r2]
                        old_dist2 = current_dist[r2]
                        # find best insertion position in route2
                        best_cost = float('inf')
                        best_pos = -1
                        for pos in range(1, len(route2)):
                            cost = (distance_matrix[route2[pos-1], cust] +
                                    distance_matrix[cust, route2[pos]] -
                                    distance_matrix[route2[pos-1], route2[pos]])
                            if cost < best_cost - 1e-9:
                                best_cost = cost
                                best_pos = pos
                        new_route2 = route2[:best_pos] + [cust] + route2[best_pos:]
                        new_dist2 = old_dist2 + best_cost
                        other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1, r2)]
                        new_max = max(new_dist1, new_dist2, *other_dists)
                        if new_max < local_best_max - 1e-9:
                            routes[r1] = new_route1
                            routes[r2] = new_route2
                            current_dist[r1] = new_dist1
                            current_dist[r2] = new_dist2
                            local_best_max = new_max
                            local_best_routes = [list(r) for r in routes]
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
                            if new_max < local_best_max - 1e-9:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                local_best_max = new_max
                                local_best_routes = [list(r) for r in routes]
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
                best_improve = 0.0
                best_i = best_j = -1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        improvement = current_dist[r] - new_dist
                        if improvement > best_improve + 1e-9:
                            best_improve = improvement
                            best_i, best_j = i, j
                if best_improve > 1e-9:
                    new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                    routes[r] = new_route
                    current_dist[r] = route_distance(new_route)
                    new_max = max(current_dist)
                    if new_max < local_best_max - 1e-9:
                        local_best_max = new_max
                        local_best_routes = [list(r) for r in routes]
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
                            if new_max < local_best_max - 1e-9:
                                routes[r1] = new1
                                routes[r2] = new2
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                local_best_max = new_max
                                local_best_routes = [list(r) for r in routes]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Update global best
        if local_best_max < best_max - 1e-9:
            best_max = local_best_max
            best_routes = [list(r) for r in local_best_routes]
            # report_best_vrp(best_routes)
    
    return best_routes