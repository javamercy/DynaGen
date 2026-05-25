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
    
    def insert_cost(route, node, current_dist, max_dist):
        best_cost = float('inf')
        best_pos = -1
        for i in range(1, len(route)):
            cost = distance_matrix[route[i-1], node] + distance_matrix[node, route[i]] - distance_matrix[route[i-1], route[i]]
            # Add balancing penalty: weight 0.3 * (current_dist + cost) / max_dist, capped to avoid extreme
            penalty = 0.3 * (current_dist + cost) / (max_dist + 1e-9)
            total = cost + penalty
            if total < best_cost:
                best_cost = total
                best_pos = i
        return best_cost, best_pos
    
    global_best_max = float('inf')
    global_best_routes = None
    num_restarts = 10
    
    for restart in range(num_restarts):
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        current_dist = [0.0 for _ in range(truck_count)]
        
        # Regret-2 insertion with balanced cost
        for _ in range(len(customers)):
            max_dist = max(current_dist) if max(current_dist) > 0 else 1.0
            best_regret = -float('inf')
            best_cust = None
            best_route = -1
            best_pos = -1
            for cust in customers:
                costs = []
                for r in range(truck_count):
                    cost, pos = insert_cost(routes[r], cust, current_dist[r], max_dist)
                    costs.append((cost, pos, r))
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
            # Insert best_cust
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
                        cost, pos = insert_cost(route2, cust, current_dist[r2], best_max)  # use best_max as reference
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
        
        # Post-optimization balancing: try to move customers from longest route to shorter ones
        max_bal_iters = 5 * n_cust
        for _ in range(max_bal_iters):
            # identify current longest route
            max_dist = max(current_dist)
            longest_routes = [r for r, d in enumerate(current_dist) if d == max_dist]
            if not longest_routes:
                break
            r_long = longest_routes[0]
            route_long = routes[r_long]
            if len(route_long) <= 2:
                break
            # try to relocate each customer from longest to another route if it reduces max
            moved = False
            for idx in range(1, len(route_long)-1):
                cust = route_long[idx]
                new_route_long = route_long[:idx] + route_long[idx+1:]
                new_dist_long = route_distance(new_route_long)
                for r_other in range(truck_count):
                    if r_other == r_long:
                        continue
                    route_other = routes[r_other]
                    cost, pos = insert_cost(route_other, cust, current_dist[r_other], best_max)
                    new_dist_other = current_dist[r_other] + cost
                    other_dists = [current_dist[i] for i in range(truck_count) if i not in (r_long, r_other)]
                    new_max = max(new_dist_long, new_dist_other, *other_dists)
                    if new_max < best_max:
                        routes[r_long] = new_route_long
                        routes[r_other] = route_other[:pos] + [cust] + route_other[pos:]
                        current_dist[r_long] = new_dist_long
                        current_dist[r_other] = new_dist_other
                        best_max = new_max
                        if new_max < global_best_max:
                            global_best_max = new_max
                            global_best_routes = [list(r) for r in routes]
                            report_best_vrp(routes)
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break
    
    return global_best_routes