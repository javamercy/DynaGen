import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        routes += [[0, 0]] * (truck_count - (n - 1))
        return routes
    
    # Step 1: Deterministic farthest-point clustering
    seed_customers = []
    # first seed: farthest from depot (no noise)
    farest = max(customers, key=lambda c: distance_matrix[0][c])
    seed_customers.append(farest)
    while len(seed_customers) < truck_count:
        best_cust = None
        best_score = -1.0
        for c in customers:
            if c in seed_customers:
                continue
            min_dist = min(distance_matrix[c][s] for s in seed_customers)
            if min_dist > best_score:
                best_score = min_dist
                best_cust = c
        seed_customers.append(best_cust)
    
    routes = [[0, s, 0] for s in seed_customers]
    route_dist = [distance_matrix[0][s] + distance_matrix[s][0] for s in seed_customers]
    assigned = set(seed_customers)
    
    # Insert remaining customers greedily by minimizing max route distance (no noise)
    remaining = [c for c in customers if c not in assigned]
    remaining.sort(key=lambda c: distance_matrix[0][c], reverse=True)
    for cust in remaining:
        best_new_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                new_route_dist = route_dist[idx] - distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
                new_overall_max = max(route_dist[:idx] + [new_route_dist] + route_dist[idx+1:])
                if new_overall_max < best_new_max:
                    best_new_max = new_overall_max
                    best_route_idx = idx
                    best_pos = pos
        route = routes[best_route_idx]
        pos = best_pos
        prev = route[pos-1]
        nxt = route[pos]
        route_dist[best_route_idx] += -distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
        route.insert(pos, cust)
        assigned.add(cust)
    
    def route_distance(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i]][route[i+1]]
        return dist
    
    best_routes = [r[:] for r in routes]
    best_max = max(route_distance(r) for r in routes)
    
    # Step 2: Intra-route 2-opt improvement (deterministic)
    max_iter = max(50, n * 2)
    for _ in range(max_iter):
        improved = False
        for idx, route in enumerate(routes):
            if len(route) <= 4:
                continue
            best_improvement = 0.0
            best_i = -1
            best_j = -1
            best_new_route = None
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j == i+1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    old_dist = route_distance(route)
                    if old_dist - new_dist > best_improvement:
                        best_improvement = old_dist - new_dist
                        best_i = i
                        best_j = j
                        best_new_route = new_route
            if best_improvement > 0:
                routes[idx] = best_new_route
                improved = True
                current_max = max(route_distance(r) for r in routes)
                if current_max < best_max:
                    best_routes = [r[:] for r in routes]
                    best_max = current_max
        if not improved:
            break
    
    # Step 3: Inter-route best-improvement (deterministic, only strict max reduction)
    max_outer_iter = max(50, n * 2)
    for outer in range(max_outer_iter):
        local_routes = [r[:] for r in best_routes]
        local_dists = [route_distance(r) for r in local_routes]
        initial_max = max(local_dists)
        best_delta = 0.0
        best_move = None
        
        # Relocations
        for i in range(truck_count):
            if len(local_routes[i]) <= 3:
                continue
            for cust in local_routes[i][1:-1]:
                route_i_new = [c for c in local_routes[i] if c != cust]
                route_i_new_dist = route_distance(route_i_new)
                for j in range(truck_count):
                    if i == j:
                        continue
                    for pos in range(1, len(local_routes[j])):
                        prev = local_routes[j][pos-1]
                        nxt = local_routes[j][pos]
                        route_j_new = local_routes[j][:pos] + [cust] + local_routes[j][pos:]
                        route_j_new_dist = route_distance(route_j_new)
                        new_dists = local_dists[:]
                        new_dists[i] = route_i_new_dist
                        new_dists[j] = route_j_new_dist
                        new_max = max(new_dists)
                        if new_max < initial_max:
                            delta = initial_max - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('relocate', i, j, cust, pos, route_i_new, route_j_new)
        
        # Exchanges
        for i in range(truck_count):
            if len(local_routes[i]) <= 3:
                continue
            for ci in local_routes[i][1:-1]:
                for j in range(i+1, truck_count):
                    if len(local_routes[j]) <= 3:
                        continue
                    for cj in local_routes[j][1:-1]:
                        route_i_new = [c if c != ci else cj for c in local_routes[i]]
                        route_j_new = [c if c != cj else ci for c in local_routes[j]]
                        route_i_new_dist = route_distance(route_i_new)
                        route_j_new_dist = route_distance(route_j_new)
                        new_dists = local_dists[:]
                        new_dists[i] = route_i_new_dist
                        new_dists[j] = route_j_new_dist
                        new_max = max(new_dists)
                        if new_max < initial_max:
                            delta = initial_max - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('exchange', i, j, ci, cj, route_i_new, route_j_new)
        
        if best_move is not None and best_delta > 0:
            if best_move[0] == 'relocate':
                _, i, j, cust, pos, route_i_new, route_j_new = best_move
                best_routes[i] = route_i_new
                best_routes[j] = route_j_new
            else:
                _, i, j, ci, cj, route_i_new, route_j_new = best_move
                best_routes[i] = route_i_new
                best_routes[j] = route_j_new
            new_max = max(route_distance(r) for r in best_routes)
            if new_max < best_max:
                best_max = new_max
        else:
            break
    
    return best_routes[:truck_count]