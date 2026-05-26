import numpy as np
import random
import math

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

def rebalance(routes, dm, truck_count, n):
    dists = [route_distance(r, dm) for r in routes]
    max_idx = max(range(truck_count), key=lambda i: dists[i])
    min_idx = min(range(truck_count), key=lambda i: dists[i])
    if dists[max_idx] <= dists[min_idx] + 1e-9:
        return False
    # try to move a customer from max route to min route to reduce max
    max_route = routes[max_idx]
    for pos in range(1, len(max_route)-1):
        cust = max_route[pos]
        new_max_route = max_route[:pos] + max_route[pos+1:]
        new_max_dist = route_distance(new_max_route, dm)
        min_route = routes[min_idx]
        for ins in range(1, len(min_route)):
            new_min_route = min_route[:ins] + [cust] + min_route[ins:]
            new_min_dist = route_distance(new_min_route, dm)
            if max(new_max_dist, new_min_dist) < dists[max_idx] - 1e-9:
                routes[max_idx] = new_max_route
                routes[min_idx] = new_min_route
                return True
    return False

def perturb_and_rebalance(routes, truck_count, n, dm):
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
        ins_pos = random.randint(1, len(route)-1) if len(route) > 2 else 1
        routes[r_idx] = route[:ins_pos] + [cust] + route[ins_pos:]
    # greedy rebalance attempts
    for _ in range(n):
        if not rebalance(routes, dm, truck_count, n):
            break

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    if truck_count >= len(customers):
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # Initialization
    routes = [[0, c, 0] for c in customers]
    
    # Clarke-Wright savings
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
    
    # Fallback merge
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
                last_i = ri[-2]
                first_i = ri[1]
                last_j = rj[-2]
                first_j = rj[1]
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
    
    dists = [route_distance(r, distance_matrix) for r in routes]
    best_routes = [list(r) for r in routes]
    best_max = max(dists)
    report_best_vrp(best_routes)
    
    current_routes = [list(r) for r in routes]
    current_max = best_max
    
    max_restarts = min(5, n)
    T0 = max(dists) * 0.1
    T = T0
    cooling_rate = 0.95
    
    for restart in range(max_restarts):
        if restart > 0:
            perturb_and_rebalance(current_routes, truck_count, n, distance_matrix)
            dists = [route_distance(r, distance_matrix) for r in current_routes]
            current_max = max(dists)
        
        max_iter = n * truck_count
        for iteration in range(max_iter):
            # local search: relocate and swap from longest route
            dists = [route_distance(r, distance_matrix) for r in current_routes]
            max_idx = max(range(truck_count), key=lambda i: dists[i])
            improved = False
            
            # Relocate
            if len(current_routes[max_idx]) > 2:
                for pos in range(1, len(current_routes[max_idx])-1):
                    cust = current_routes[max_idx][pos]
                    new_max_route = current_routes[max_idx][:pos] + current_routes[max_idx][pos+1:]
                    new_max_dist = route_distance(new_max_route, distance_matrix)
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = current_routes[other_idx]
                        for ins in range(1, len(other_route)):
                            new_other_route = other_route[:ins] + [cust] + other_route[ins:]
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            delta = new_max - current_max
                            if delta < 0 or (T > 1e-6 and random.random() < math.exp(-delta/T)):
                                current_routes[max_idx] = new_max_route
                                current_routes[other_idx] = new_other_route
                                current_max = new_max
                                if new_max < best_max - 1e-9:
                                    best_max = new_max
                                    best_routes = [list(r) for r in current_routes]
                                    report_best_vrp(best_routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Swap
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
                            delta = new_max - current_max
                            if delta < 0 or (T > 1e-6 and random.random() < math.exp(-delta/T)):
                                current_routes[max_idx] = new_max_route
                                current_routes[other_idx] = new_other_route
                                current_max = new_max
                                if new_max < best_max - 1e-9:
                                    best_max = new_max
                                    best_routes = [list(r) for r in current_routes]
                                    report_best_vrp(best_routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Always apply 2-opt on longest route
            if len(current_routes[max_idx]) > 2:
                new_route = two_opt(current_routes[max_idx], distance_matrix)
                new_dist = route_distance(new_route, distance_matrix)
                if new_dist < dists[max_idx] - 1e-9:
                    current_routes[max_idx] = new_route
                    current_max = max(route_distance(r, distance_matrix) for r in current_routes)
                    if current_max < best_max - 1e-9:
                        best_max = current_max
                        best_routes = [list(r) for r in current_routes]
                        report_best_vrp(best_routes)
                    improved = True
            
            if not improved:
                break
        
        # Cool down temperature for next restart
        T *= cooling_rate
    
    report_best_vrp(best_routes)
    return best_routes