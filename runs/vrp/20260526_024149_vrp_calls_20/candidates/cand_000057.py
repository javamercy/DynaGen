import numpy as np
import random

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def two_opt(route, dm):
    improved = True
    best_route = route[:]
    best_dist = route_distance(best_route, dm)
    while improved:
        improved = False
        for i in range(1, len(best_route)-2):
            for j in range(i+1, len(best_route)-1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_dist = route_distance(new_route, dm)
                if new_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    return best_route

def perturb(routes, truck_count, n):
    num_cust = min(3, n-1)
    customers = list(range(1, n))
    random.shuffle(customers)
    selected = customers[:num_cust]
    removal_list = []
    for cust in selected:
        for idx, route in enumerate(routes):
            if cust in route:
                pos = route.index(cust)
                removal_list.append((idx, pos))
                break
    removal_list.sort(key=lambda x: -x[1])
    for idx, pos in removal_list:
        routes[idx] = routes[idx][:pos] + routes[idx][pos+1:]
    for cust in selected:
        r_idx = random.randint(0, truck_count-1)
        route = routes[r_idx]
        if len(route) <= 2:
            insert_pos = 1
        else:
            insert_pos = random.randint(1, len(route)-1)
        routes[r_idx] = route[:insert_pos] + [cust] + route[insert_pos:]
    return routes

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    if truck_count >= len(customers):
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # Clarke-Wright savings merging
    best_routes = None
    best_max = float('inf')
    num_restarts = min(3, n * truck_count // 10 + 1)
    
    for restart in range(num_restarts):
        random.seed(restart)  # Different seed for each restart
        # Initialization: each customer as its own route
        routes = [[0, c, 0] for c in customers]
        
        while len(routes) > truck_count:
            best_saving = -1e9
            best_pair = None
            best_order = 0
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    ri = routes[i]
                    rj = routes[j]
                    if len(ri) <= 2 or len(rj) <= 2:
                        continue
                    last_i = ri[-2]
                    first_i = ri[1]
                    last_j = rj[-2]
                    first_j = rj[1]
                    s1 = distance_matrix[0][last_i] + distance_matrix[0][first_j] - distance_matrix[last_i][first_j]
                    s2 = distance_matrix[0][last_j] + distance_matrix[0][first_i] - distance_matrix[last_j][first_i]
                    if s1 > best_saving:
                        best_saving = s1
                        best_pair = (i, j)
                        best_order = 0
                    if s2 > best_saving:
                        best_saving = s2
                        best_pair = (i, j)
                        best_order = 1
            if best_pair is None:
                break
            i, j = best_pair
            if best_order == 0:
                new_route = routes[i][:-1] + routes[j][1:]
            else:
                new_route = routes[j][:-1] + routes[i][1:]
            if i < j:
                del routes[j]
                del routes[i]
            else:
                del routes[i]
                del routes[j]
            routes.append(new_route)
        
        # Fallback merging
        while len(routes) > truck_count:
            best_pair = None
            best_merge_route = None
            best_new_max = float('inf')
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    ri = routes[i]
                    rj = routes[j]
                    if len(ri) <= 2 or len(rj) <= 2:
                        continue
                    for order in [0, 1]:
                        if order == 0:
                            new_route = ri[:-1] + rj[1:]
                        else:
                            new_route = rj[:-1] + ri[1:]
                        new_dists = [route_distance(r, distance_matrix) for r in routes]
                        new_dists[i] = route_distance(new_route, distance_matrix)
                        new_dists[j] = 0
                        new_max = max(new_dists)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_pair = (i, j)
                            best_order = order
                            best_merge_route = new_route
            if best_pair is None:
                break
            i, j = best_pair
            new_route = best_merge_route
            if i < j:
                del routes[j]
                del routes[i]
            else:
                del routes[i]
                del routes[j]
            routes.append(new_route)
        
        current_routes = [list(r) for r in routes]
        max_iter = n * truck_count
        
        for iteration in range(max_iter):
            dists = [route_distance(r, distance_matrix) for r in current_routes]
            max_dist = max(dists)
            if max_dist < best_max - 1e-9:
                best_max = max_dist
                best_routes = [list(r) for r in current_routes]
                report_best_vrp(best_routes)
            
            max_idx = dists.index(max_dist)
            improved = False
            
            # Relocate from longest route
            if len(current_routes[max_idx]) > 2:
                for pos in range(1, len(current_routes[max_idx])-1):
                    cust = current_routes[max_idx][pos]
                    new_max_route = current_routes[max_idx][:pos] + current_routes[max_idx][pos+1:]
                    new_max_dist = route_distance(new_max_route, distance_matrix)
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = current_routes[other_idx]
                        for insert_pos in range(1, len(other_route)):
                            new_other_route = other_route[:insert_pos] + [cust] + other_route[insert_pos:]
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            if new_max < max_dist - 1e-9:
                                current_routes[max_idx] = new_max_route
                                current_routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # If no relocate improvement, try swap
            if not improved and len(current_routes[max_idx]) > 2:
                for other_idx in range(truck_count):
                    if other_idx == max_idx or len(current_routes[other_idx]) <= 2:
                        continue
                    for pos_max in range(1, len(current_routes[max_idx])-1):
                        cust_a = current_routes[max_idx][pos_max]
                        for pos_other in range(1, len(current_routes[other_idx])-1):
                            cust_b = current_routes[other_idx][pos_other]
                            new_max_route = current_routes[max_idx].copy()
                            new_max_route[pos_max] = cust_b
                            new_max_dist = route_distance(new_max_route, distance_matrix)
                            new_other_route = current_routes[other_idx].copy()
                            new_other_route[pos_other] = cust_a
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            if new_max < max_dist - 1e-9:
                                current_routes[max_idx] = new_max_route
                                current_routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Always apply 2-opt on the current longest route
            dists = [route_distance(r, distance_matrix) for r in current_routes]
            max_idx = dists.index(max(dists))
            if len(current_routes[max_idx]) > 2:
                new_route = two_opt(current_routes[max_idx], distance_matrix)
                new_dist = route_distance(new_route, distance_matrix)
                if new_dist < dists[max_idx] - 1e-9:
                    current_routes[max_idx] = new_route
                    improved = True
            
            if not improved:
                break
        
        # Final check for this restart
        final_dists = [route_distance(r, distance_matrix) for r in current_routes]
        current_max = max(final_dists)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
    
    report_best_vrp(best_routes)
    return best_routes