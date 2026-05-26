import numpy as np
import random
from collections import defaultdict

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
    num_restarts = 3
    max_perturbations = 2

    for restart in range(num_restarts):
        random.shuffle(customers)
        # Clarke-Wright construction
        # Initialize each customer as a separate route
        routes = [[0, c, 0] for c in customers]
        # Compute savings
        savings = []
        for i in customers:
            for j in customers:
                if i < j:
                    s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                    savings.append((s, i, j))
        savings.sort(reverse=True, key=lambda x: x[0])
        
        # While more routes than truck_count
        route_map = {c: idx for idx, r in enumerate(routes) for c in r[1:-1]}
        # Track current routes list
        for s, i, j in savings:
            if len(routes) <= truck_count:
                break
            if i not in route_map or j not in route_map:
                continue
            ri = route_map[i]
            rj = route_map[j]
            if ri == rj:
                continue
            # Check if i and j are endpoints
            route_i = routes[ri]
            route_j = routes[rj]
            # Ensure i is last internal or j is first internal
            # For merging, we need i to be last in its route (before depot) and j to be first after depot
            if route_i[-2] == i and route_j[1] == j:
                # merge route_i + route_j[1:] but careful: route_i ends with 0, route_j starts with 0
                new_route = route_i[:-1] + route_j[1:]
            elif route_i[1] == i and route_j[-2] == j:
                new_route = route_j[:-1] + route_i[1:]
            else:
                continue
            # Merge
            # Remove both routes, add new
            if ri < rj:
                del routes[rj]
                del routes[ri]
            else:
                del routes[ri]
                del routes[rj]
            routes.append(new_route)
            # Update route_map
            route_map = {c: idx for idx, r in enumerate(routes) for c in r[1:-1]}
        
        # Ensure exactly truck_count routes: if fewer, add empty; if more, need to merge remaining? Actually should have exactly truck_count after merging, but if not, fill with empty
        while len(routes) < truck_count:
            routes.append([0, 0])
        if len(routes) > truck_count:
            # Should not happen, but merge some routes arbitrarily
            while len(routes) > truck_count:
                # merge last two routes
                r1 = routes.pop()
                r2 = routes.pop()
                new_route = r1[:-1] + r2[1:]
                routes.append(new_route)
        
        # Intra-route 2-opt on each route
        for idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            improved = True
            iters = 0
            max_iters = len(route) * len(route)
            while improved and iters < max_iters:
                improved = False
                iters += 1
                best_delta = 0
                best_ij = None
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route, distance_matrix)
                        old_dist = route_distance(route, distance_matrix)
                        delta = old_dist - new_dist
                        if delta > best_delta + 1e-12:
                            best_delta = delta
                            best_ij = (i, j)
                            improved = True
                if improved:
                    i, j = best_ij
                    route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    routes[idx] = route
            routes[idx] = route
        
        # Main improvement loop with early termination
        def inter_improve(routes, max_no_improve=50):
            no_improve = 0
            max_iter = num_customers * truck_count * 2
            for _ in range(max_iter):
                if no_improve >= max_no_improve:
                    break
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
                # 2-opt*
                for i in range(len(routes)):
                    if len(routes[i]) <= 3:
                        continue
                    for j in range(i+1, len(routes)):
                        if len(routes[j]) <= 3:
                            continue
                        for p1 in range(1, len(routes[i])-1):
                            for p2 in range(1, len(routes[j])-1):
                                new_route_i = routes[i][:p1] + routes[j][p2:]
                                new_route_j = routes[j][:p2] + routes[i][p1:]
                                if new_route_i[0] != 0 or new_route_i[-1] != 0 or new_route_j[0] != 0 or new_route_j[-1] != 0:
                                    continue
                                new_dists = dists.copy()
                                new_dists[i] = route_distance(new_route_i, distance_matrix)
                                new_dists[j] = route_distance(new_route_j, distance_matrix)
                                new_max = max(new_dists)
                                improvement = max_dist - new_max
                                if improvement > best_improvement + 1e-12:
                                    best_improvement = improvement
                                    best_move = ('cross', i, p1, j, p2)
                if best_improvement > 1e-9:
                    typ = best_move[0]
                    if typ == 'reloc':
                        _, r1, p1, r2, p2 = best_move
                        cust = routes[r1][p1]
                        routes[r1] = routes[r1][:p1] + routes[r1][p1+1:]
                        routes[r2] = routes[r2][:p2] + [cust] + routes[r2][p2:]
                    elif typ == 'swap':
                        _, r1, p1, r2, p2 = best_move
                        cust_i = routes[r1][p1]
                        cust_j = routes[r2][p2]
                        routes[r1] = routes[r1][:p1] + [cust_j] + routes[r1][p1+1:]
                        routes[r2] = routes[r2][:p2] + [cust_i] + routes[r2][p2+1:]
                    elif typ == 'cross':
                        _, r1, p1, r2, p2 = best_move
                        new_r1 = routes[r1][:p1] + routes[r2][p2:]
                        new_r2 = routes[r2][:p2] + routes[r1][p1:]
                        routes[r1] = new_r1
                        routes[r2] = new_r2
                    no_improve = 0
                    report_best_vrp(routes)
                else:
                    no_improve += 1
            return routes
        
        routes = inter_improve(routes, max_no_improve=50)
        
        # Perturbation loop
        for pert in range(max_perturbations):
            # Eject random customers
            current_customers = []
            for r in routes:
                current_customers.extend(r[1:-1])
            if len(current_customers) < 3:
                break
            k = random.randint(1, min(3, len(current_customers)))
            ejected = random.sample(current_customers, k)
            # Remove ejected
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                for cust in ejected:
                    while cust in route:
                        route.remove(cust)
                if len(route) == 2:
                    routes[r_idx] = [0, 0]
                else:
                    routes[r_idx] = route
            # Reinsert greedily
            for cust in ejected:
                best_route_idx = None
                best_pos = None
                best_new_max = float('inf')
                for r_idx in range(len(routes)):
                    route = routes[r_idx]
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_dist = route_distance(new_route, distance_matrix)
                        other_dists = [route_distance(routes[j], distance_matrix) for j in range(len(routes)) if j != r_idx]
                        new_max = max(max(other_dists) if other_dists else 0, new_dist)
                        if new_max < best_new_max - 1e-12:
                            best_new_max = new_max
                            best_route_idx = r_idx
                            best_pos = pos
                if best_route_idx is not None:
                    routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
            report_best_vrp(routes)
            # Re-apply improvement
            routes = inter_improve(routes, max_no_improve=20)
        
        current_max = compute_max_dist(routes, distance_matrix)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
    
    if best_routes is not None:
        routes = best_routes
    report_best_vrp(routes)
    return routes