import numpy as np
import random


def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))


def compute_max_dist(routes, dm):
    return max(route_distance(r, dm) for r in routes)


def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    num_customers = len(customers)
    
    if truck_count >= num_customers:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    best_routes = None
    best_max = float('inf')
    
    # Multi-start with different customer orderings
    orders = [
        customers[:],
        customers[::-1],
        sorted(customers, key=lambda x: distance_matrix[0, x])
    ]
    # Add a random seed order deterministically
    random.seed(0)
    shuffled = customers[:]
    random.shuffle(shuffled)
    orders.append(shuffled)
    
    for order in orders[:3]:  # Use first 3 distinct orders
        # Build initial routes from order
        routes = [[0, c, 0] for c in order]
        
        while len(routes) > truck_count:
            best_saving = -1e9
            best_pair = None
            best_order = 0
            best_trad = -1e9
            current_max = compute_max_dist(routes, distance_matrix)
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    r_i = routes[i]
                    r_j = routes[j]
                    if len(r_i) <= 2 or len(r_j) <= 2:
                        continue
                    for order_flag in [0, 1]:
                        if order_flag == 0:
                            new_route = r_i[:-1] + r_j[1:]
                        else:
                            new_route = r_j[:-1] + r_i[1:]
                        
                        # Compute new max distance
                        other_dists = [route_distance(routes[k], distance_matrix) for k in range(len(routes)) if k != i and k != j]
                        new_dist = route_distance(new_route, distance_matrix)
                        new_max = max(max(other_dists) if other_dists else 0, new_dist)
                        saving = current_max - new_max
                        
                        # Balancing term: discourage merging routes with very different lengths
                        dist_i = route_distance(r_i, distance_matrix)
                        dist_j = route_distance(r_j, distance_matrix)
                        balancing = 0.1 * abs(dist_i - dist_j)
                        saving -= balancing
                        
                        # Traditional Clarke-Wright savings for tie-breaking
                        last_i = r_i[-2]
                        first_i = r_i[1]
                        last_j = r_j[-2]
                        first_j = r_j[1]
                        if order_flag == 0:
                            trad_saving = distance_matrix[0, last_i] + distance_matrix[0, first_j] - distance_matrix[last_i, first_j]
                        else:
                            trad_saving = distance_matrix[0, last_j] + distance_matrix[0, first_i] - distance_matrix[last_j, first_i]
                        
                        if saving > best_saving + 1e-12 or (abs(saving - best_saving) < 1e-12 and trad_saving > best_trad + 1e-12):
                            best_saving = saving
                            best_trad = trad_saving
                            best_pair = (i, j)
                            best_order = order_flag
            
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
        
        # Intra-route 2-opt
        for idx in range(len(routes)):
            route = routes[idx]
            if len(route) <= 3:
                continue
            improved = True
            max_iter = len(route) * len(route)
            it = 0
            while improved and it < max_iter:
                improved = False
                it += 1
                best_delta = 0
                best_ij = None
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route, distance_matrix)
                        old_dist = route_distance(route, distance_matrix)
                        delta = old_dist - new_dist
                        if delta > best_delta:
                            best_delta = delta
                            best_ij = (i, j)
                            improved = True
                if improved:
                    i, j = best_ij
                    route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    routes[idx] = route
            routes[idx] = route
        
        # Inter-route improvement with perturbation
        max_iter = num_customers * truck_count * 2
        perturbation_count = 0
        max_perturbations = 5
        
        while True:
            improved_global = False
            for _ in range(max_iter):
                dists = [route_distance(r, distance_matrix) for r in routes]
                max_dist = max(dists)
                best_improvement = 0
                best_move = None
                # Relocate
                for i in range(len(routes)):
                    if len(routes[i]) <= 2:
                        continue
                    for pos in range(1, len(routes[i])-1):
                        customer = routes[i][pos]
                        for j in range(len(routes)):
                            if i == j:
                                continue
                            for insert_pos in range(1, len(routes[j])):
                                new_route_i = routes[i][:pos] + routes[i][pos+1:]
                                new_route_j = routes[j][:insert_pos] + [customer] + routes[j][insert_pos:]
                                new_dists = dists.copy()
                                new_dists[i] = route_distance(new_route_i, distance_matrix)
                                new_dists[j] = route_distance(new_route_j, distance_matrix)
                                new_max = max(new_dists)
                                improvement = max_dist - new_max
                                if improvement > best_improvement + 1e-12:
                                    best_improvement = improvement
                                    best_move = ('reloc', i, pos, j, insert_pos)
                # Swap
                for i in range(len(routes)):
                    if len(routes[i]) <= 2:
                        continue
                    for pos_i in range(1, len(routes[i])-1):
                        cust_i = routes[i][pos_i]
                        for j in range(i+1, len(routes)):
                            if len(routes[j]) <= 2:
                                continue
                            for pos_j in range(1, len(routes[j])-1):
                                cust_j = routes[j][pos_j]
                                new_route_i = routes[i][:pos_i] + [cust_j] + routes[i][pos_i+1:]
                                new_route_j = routes[j][:pos_j] + [cust_i] + routes[j][pos_j+1:]
                                new_dists = dists.copy()
                                new_dists[i] = route_distance(new_route_i, distance_matrix)
                                new_dists[j] = route_distance(new_route_j, distance_matrix)
                                new_max = max(new_dists)
                                improvement = max_dist - new_max
                                if improvement > best_improvement + 1e-12:
                                    best_improvement = improvement
                                    best_move = ('swap', i, pos_i, j, pos_j)
                if best_improvement > 1e-9:
                    typ, r1, p1, r2, p2 = best_move
                    if typ == 'reloc':
                        cust = routes[r1][p1]
                        routes[r1] = routes[r1][:p1] + routes[r1][p1+1:]
                        routes[r2] = routes[r2][:p2] + [cust] + routes[r2][p2:]
                    else:
                        cust_i = routes[r1][p1]
                        cust_j = routes[r2][p2]
                        routes[r1] = routes[r1][:p1] + [cust_j] + routes[r1][p1+1:]
                        routes[r2] = routes[r2][:p2] + [cust_i] + routes[r2][p2+1:]
                    improved_global = True
                    report_best_vrp(routes)
                else:
                    break
            
            if not improved_global:
                if perturbation_count >= max_perturbations:
                    break
                perturbation_count += 1
                best_worsen = 1e9
                best_move = None
                dists = [route_distance(r, distance_matrix) for r in routes]
                max_dist = max(dists)
                # Relocate
                for i in range(len(routes)):
                    if len(routes[i]) <= 2:
                        continue
                    for pos in range(1, len(routes[i])-1):
                        customer = routes[i][pos]
                        for j in range(len(routes)):
                            if i == j:
                                continue
                            for insert_pos in range(1, len(routes[j])):
                                new_route_i = routes[i][:pos] + routes[i][pos+1:]
                                new_route_j = routes[j][:insert_pos] + [customer] + routes[j][insert_pos:]
                                new_dists = dists.copy()
                                new_dists[i] = route_distance(new_route_i, distance_matrix)
                                new_dists[j] = route_distance(new_route_j, distance_matrix)
                                new_max = max(new_dists)
                                worsen = new_max - max_dist
                                if worsen < best_worsen - 1e-12:
                                    best_worsen = worsen
                                    best_move = ('reloc', i, pos, j, insert_pos)
                # Swap
                for i in range(len(routes)):
                    if len(routes[i]) <= 2:
                        continue
                    for pos_i in range(1, len(routes[i])-1):
                        cust_i = routes[i][pos_i]
                        for j in range(i+1, len(routes)):
                            if len(routes[j]) <= 2:
                                continue
                            for pos_j in range(1, len(routes[j])-1):
                                cust_j = routes[j][pos_j]
                                new_route_i = routes[i][:pos_i] + [cust_j] + routes[i][pos_i+1:]
                                new_route_j = routes[j][:pos_j] + [cust_i] + routes[j][pos_j+1:]
                                new_dists = dists.copy()
                                new_dists[i] = route_distance(new_route_i, distance_matrix)
                                new_dists[j] = route_distance(new_route_j, distance_matrix)
                                new_max = max(new_dists)
                                worsen = new_max - max_dist
                                if worsen < best_worsen - 1e-12:
                                    best_worsen = worsen
                                    best_move = ('swap', i, pos_i, j, pos_j)
                if best_move is None or best_worsen >= 1e9:
                    break
                typ, r1, p1, r2, p2 = best_move
                if typ == 'reloc':
                    cust = routes[r1][p1]
                    routes[r1] = routes[r1][:p1] + routes[r1][p1+1:]
                    routes[r2] = routes[r2][:p2] + [cust] + routes[r2][p2:]
                else:
                    cust_i = routes[r1][p1]
                    cust_j = routes[r2][p2]
                    routes[r1] = routes[r1][:p1] + [cust_j] + routes[r1][p1+1:]
                    routes[r2] = routes[r2][:p2] + [cust_i] + routes[r2][p2+1:]
                report_best_vrp(routes)
            else:
                perturbation_count = 0
                continue
        
        # Update best solution
        current_max = compute_max_dist(routes, distance_matrix)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
    
    # Ensure exactly truck_count routes
    if len(best_routes) != truck_count:
        # Should not happen, but if so, adjust
        while len(best_routes) < truck_count:
            best_routes.append([0, 0])
    
    report_best_vrp(best_routes)
    return best_routes