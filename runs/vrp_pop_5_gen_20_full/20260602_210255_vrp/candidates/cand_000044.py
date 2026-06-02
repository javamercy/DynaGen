import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        routes += [[0, 0]] * (truck_count - (n - 1))
        return routes
    
    def route_distance(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    
    def compute_distances(routes):
        return [route_distance(r) for r in routes]
    
    def copy_routes(routes):
        return [r[:] for r in routes]
    
    # Farthest-point seeding
    seed_customers = []
    first_seed = max(range(1, n), key=lambda c: distance_matrix[0][c])
    seed_customers.append(first_seed)
    while len(seed_customers) < truck_count:
        best_cust = None
        best_min_dist = -1
        for c in range(1, n):
            if c in seed_customers:
                continue
            min_dist = min(distance_matrix[c][s] for s in seed_customers)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_cust = c
        seed_customers.append(best_cust)
    
    routes = [[0, s, 0] for s in seed_customers]
    route_dists = [distance_matrix[0][s] + distance_matrix[s][0] for s in seed_customers]
    assigned = set(seed_customers)
    remaining = [c for c in range(1, n) if c not in assigned]
    
    # Regret insertion
    while remaining:
        best_regret = -1
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_new_max = None
        for cust in remaining:
            insertion_costs = []
            for idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    next = route[pos]
                    new_dist = route_dists[idx] - distance_matrix[prev][next] + distance_matrix[prev][cust] + distance_matrix[cust][next]
                    overall_max = max(route_dists[:idx] + [new_dist] + route_dists[idx+1:])
                    insertion_costs.append((overall_max, idx, pos))
            insertion_costs.sort(key=lambda x: x[0])
            best_cost = insertion_costs[0][0]
            if len(insertion_costs) > 1:
                regret = insertion_costs[1][0] - best_cost
            else:
                regret = best_cost
            if regret > best_regret or (regret == best_regret and best_cust is None):
                best_regret = regret
                best_cust = cust
                best_new_max, best_route_idx, best_pos = insertion_costs[0]
        
        route = routes[best_route_idx]
        pos = best_pos
        prev = route[pos-1]
        next = route[pos]
        route_dists[best_route_idx] += -distance_matrix[prev][next] + distance_matrix[prev][best_cust] + distance_matrix[best_cust][next]
        route.insert(pos, best_cust)
        assigned.add(best_cust)
        remaining.remove(best_cust)
    
    best_routes = copy_routes(routes)
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    
    max_iter = max(50, n * 2)
    restart_count = 0
    max_restarts = 3
    stagnation = 0
    current_max = best_max
    
    for global_iter in range(max_iter):
        improved = False
        best_move = None
        best_new_max_val = float('inf')
        
        # Intra 2-opt
        for idx, route in enumerate(routes):
            if len(route) <= 4:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j == i+1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    new_max = max(route_dists[:idx] + [new_dist] + route_dists[idx+1:])
                    if new_max < current_max and new_max < best_new_max_val:
                        best_new_max_val = new_max
                        best_move = ('2opt', idx, i, j, new_route)
        
        # Inter relocate
        for i in range(truck_count):
            for j in range(truck_count):
                if i == j:
                    continue
                route_i = routes[i]
                route_j = routes[j]
                if len(route_i) <= 3:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust = route_i[pos_i]
                    new_i = route_i[:pos_i] + route_i[pos_i+1:]
                    new_i_dist = route_distance(new_i)
                    for pos_j in range(1, len(route_j)):
                        new_j = route_j[:pos_j] + [cust] + route_j[pos_j:]
                        new_j_dist = route_distance(new_j)
                        new_max = max(route_dists[:i] + [new_i_dist] + route_dists[i+1:j] + [new_j_dist] + route_dists[j+1:])
                        if new_max < current_max and new_max < best_new_max_val:
                            best_new_max_val = new_max
                            best_move = ('relocate', i, pos_i, j, pos_j, cust)
        
        # Inter exchange
        for i in range(truck_count):
            for j in range(i+1, truck_count):
                route_i = routes[i]
                route_j = routes[j]
                if len(route_i) <= 3 or len(route_j) <= 3:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust_i = route_i[pos_i]
                    for pos_j in range(1, len(route_j)-1):
                        cust_j = route_j[pos_j]
                        new_i = route_i[:pos_i] + [cust_j] + route_i[pos_i+1:]
                        new_j = route_j[:pos_j] + [cust_i] + route_j[pos_j:]
                        new_i_dist = route_distance(new_i)
                        new_j_dist = route_distance(new_j)
                        new_max = max(route_dists[:i] + [new_i_dist] + route_dists[i+1:j] + [new_j_dist] + route_dists[j+1:])
                        if new_max < current_max and new_max < best_new_max_val:
                            best_new_max_val = new_max
                            best_move = ('exchange', i, pos_i, j, pos_j, cust_i, cust_j)
        
        if best_move is not None:
            move_type = best_move[0]
            if move_type == '2opt':
                _, idx, i, j, new_route = best_move
                routes[idx] = new_route
                route_dists[idx] = route_distance(new_route)
            elif move_type == 'relocate':
                _, i, pos_i, j, pos_j, cust = best_move
                route_i = routes[i]
                routes[i] = route_i[:pos_i] + route_i[pos_i+1:]
                route_j = routes[j]
                routes[j] = route_j[:pos_j] + [cust] + route_j[pos_j:]
                route_dists[i] = route_distance(routes[i])
                route_dists[j] = route_distance(routes[j])
            elif move_type == 'exchange':
                _, i, pos_i, j, pos_j, cust_i, cust_j = best_move
                route_i = routes[i]
                route_j = routes[j]
                routes[i] = route_i[:pos_i] + [cust_j] + route_i[pos_i+1:]
                routes[j] = route_j[:pos_j] + [cust_i] + route_j[pos_j:]
                route_dists[i] = route_distance(routes[i])
                route_dists[j] = route_distance(routes[j])
            
            current_max = max(route_dists)
            if current_max < best_max:
                best_max = current_max
                best_routes = copy_routes(routes)
                report_best_vrp(best_routes)
            stagnation = 0
            improved = True
        else:
            stagnation += 1
            if stagnation >= 10 and restart_count < max_restarts:
                num_remove = max(1, int(0.1 * (n-1)))
                all_assigned = list(assigned)
                remove_set = set(random.sample(all_assigned, min(num_remove, len(all_assigned))))
                new_routes = []
                new_assigned = set()
                for route in routes:
                    new_route = [0]
                    for node in route[1:-1]:
                        if node not in remove_set:
                            new_route.append(node)
                            new_assigned.add(node)
                    new_route.append(0)
                    new_routes.append(new_route)
                routes = [r for r in new_routes if len(r) > 2]
                while len(routes) < truck_count:
                    routes.append([0, 0])
                route_dists = compute_distances(routes)
                assigned = new_assigned
                remaining = [c for c in range(1, n) if c not in assigned]
                while remaining:
                    best_regret = -1
                    best_cust = None
                    best_route_idx = None
                    best_pos = None
                    for cust in remaining:
                        insertion_costs = []
                        for idx, route in enumerate(routes):
                            for pos in range(1, len(route)):
                                prev = route[pos-1]
                                next = route[pos]
                                new_dist = route_dists[idx] - distance_matrix[prev][next] + distance_matrix[prev][cust] + distance_matrix[cust][next]
                                overall_max = max(route_dists[:idx] + [new_dist] + route_dists[idx+1:])
                                insertion_costs.append((overall_max, idx, pos))
                        insertion_costs.sort(key=lambda x: x[0])
                        best_cost = insertion_costs[0][0]
                        if len(insertion_costs) > 1:
                            regret = insertion_costs[1][0] - best_cost
                        else:
                            regret = best_cost
                        if regret > best_regret:
                            best_regret = regret
                            best_cust = cust
                            best_new_max, best_route_idx, best_pos = insertion_costs[0]
                    route = routes[best_route_idx]
                    pos = best_pos
                    prev = route[pos-1]
                    next = route[pos]
                    route_dists[best_route_idx] += -distance_matrix[prev][next] + distance_matrix[prev][best_cust] + distance_matrix[best_cust][next]
                    route.insert(pos, best_cust)
                    assigned.add(best_cust)
                    remaining.remove(best_cust)
                current_max = max(route_dists)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = copy_routes(routes)
                    report_best_vrp(best_routes)
                stagnation = 0
                restart_count += 1
        
        if global_iter == max_iter - 1:
            break
    
    return best_routes[:truck_count]