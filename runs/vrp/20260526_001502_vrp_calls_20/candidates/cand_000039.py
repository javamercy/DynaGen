import numpy as np
import random

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
    
    for restart in range(num_restarts):
        # Initialize routes
        routes = [[0, 0] for _ in range(truck_count)]
        current_dist = [0.0 for _ in range(truck_count)]
        customers = list(range(1, n))
        random.shuffle(customers)
        
        # Regret-2 insertion
        unassigned = set(customers)
        while unassigned:
            # Compute regret for each unassigned customer
            best_regret = -1
            best_cust = None
            best_route_for_cust = -1
            best_pos_for_cust = -1
            best_new_max_for_cust = float('inf')
            for cust in unassigned:
                # Compute insertion costs for each route
                costs = []
                for r in range(truck_count):
                    cost, pos = insert_cost(routes[r], cust)
                    costs.append((cost, pos, r))
                # Sort by cost
                costs.sort(key=lambda x: x[0])
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = costs[0][0]  # only one route
                # Determine best route based on minimizing max distance
                best_new_max = float('inf')
                best_r = -1
                best_p = -1
                for cost, pos, r in costs:
                    new_dist = current_dist[r] + cost
                    other_dists = [current_dist[i] for i in range(truck_count) if i != r]
                    new_max = max(new_dist, *other_dists)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_r = r
                        best_p = pos
                # Select customer with highest regret; tie-break by lower max?
                if regret > best_regret or (regret == best_regret and best_new_max < best_new_max_for_cust):
                    best_regret = regret
                    best_cust = cust
                    best_route_for_cust = best_r
                    best_pos_for_cust = best_p
                    best_new_max_for_cust = best_new_max
            # Insert best customer
            routes[best_route_for_cust].insert(best_pos_for_cust, best_cust)
            current_dist[best_route_for_cust] = route_distance(routes[best_route_for_cust])
            unassigned.remove(best_cust)
        
        best_max = max(current_dist)
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [list(r) for r in routes]
            # report_best_vrp would be called here; we call after finding improvement
            # report_best_vrp(routes)  # uncomment if needed in environment
        
        # Local search (same as parent, but with local best tracking)
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
                        if new_max < best_max - 1e-10:
                            routes[r1] = new_route1
                            routes[r2] = route2[:pos] + [cust] + route2[pos:]
                            current_dist[r1] = new_dist1
                            current_dist[r2] = new_dist2
                            best_max = new_max
                            if new_max < global_best_max - 1e-10:
                                global_best_max = new_max
                                global_best_routes = [list(r) for r in routes]
                                # report_best_vrp(routes)
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
                            if new_max < best_max - 1e-10:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                best_max = new_max
                                if new_max < global_best_max - 1e-10:
                                    global_best_max = new_max
                                    global_best_routes = [list(r) for r in routes]
                                    # report_best_vrp(routes)
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
                if best_improve > 1e-9:
                    new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                    routes[r] = new_route
                    current_dist[r] = route_distance(new_route)
                    new_max = max(current_dist)
                    if new_max < best_max - 1e-10:
                        best_max = new_max
                        if new_max < global_best_max - 1e-10:
                            global_best_max = new_max
                            global_best_routes = [list(r) for r in routes]
                            # report_best_vrp(routes)
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
                            if new_max < best_max - 1e-10:
                                routes[r1] = new1
                                routes[r2] = new2
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                best_max = new_max
                                if new_max < global_best_max - 1e-10:
                                    global_best_max = new_max
                                    global_best_routes = [list(r) for r in routes]
                                    # report_best_vrp(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    return global_best_routes