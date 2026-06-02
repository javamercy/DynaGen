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
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i]][route[i+1]]
        return dist
    
    def compute_route_dists(routes):
        return [route_distance(r) for r in routes]
    
    def copy_routes(routes):
        return [r[:] for r in routes]
    
    # Farthest-point seeding
    seed_customers = []
    first_seed = max(range(1, n), key=lambda c: distance_matrix[0][c])
    seed_customers.append(first_seed)
    while len(seed_customers) < truck_count:
        best_cust = None
        best_min_dist = -1.0
        for c in range(1, n):
            if c in seed_customers:
                continue
            min_dist = min(distance_matrix[c][s] for s in seed_customers)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_cust = c
        seed_customers.append(best_cust)
    
    routes = [[0, s, 0] for s in seed_customers]
    route_dists = compute_route_dists(routes)
    assigned = set(seed_customers)
    remaining = [c for c in range(1, n) if c not in assigned]
    
    # Regret insertion with best insertion minimizing max distance
    while remaining:
        best_regret = -1.0
        best_cust = None
        best_route_idx = None
        best_pos = None
        for cust in remaining:
            insertion_costs = []  # (new_max, route_idx, pos)
            for idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    new_dist = route_dists[idx] - distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
                    new_max = max(route_dists[:idx] + [new_dist] + route_dists[idx+1:])
                    insertion_costs.append((new_max, idx, pos))
            insertion_costs.sort(key=lambda x: (x[0], x[1], x[2]))
            best_cost = insertion_costs[0][0]
            regret = insertion_costs[1][0] - best_cost if len(insertion_costs) > 1 else best_cost
            if regret > best_regret or (regret == best_regret and best_cust is None):
                best_regret = regret
                best_cust = cust
                best_new_max, best_route_idx, best_pos = insertion_costs[0]
        # Insert best_cust
        route = routes[best_route_idx]
        pos = best_pos
        prev = route[pos-1]
        nxt = route[pos]
        route_dists[best_route_idx] += -distance_matrix[prev][nxt] + distance_matrix[prev][best_cust] + distance_matrix[best_cust][nxt]
        route.insert(pos, best_cust)
        assigned.add(best_cust)
        remaining.remove(best_cust)
    
    best_routes = copy_routes(routes)
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    
    # Local search with best improvement
    max_iter = max(40, n)
    restart_count = 0
    max_restarts = 1
    stagnation = 0
    
    def apply_2opt(route, i, j):
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        return new_route
    
    for _ in range(max_iter):
        improved = False
        # Best improvement intra 2-opt
        best_move = None
        best_new_max = best_max
        for idx, route in enumerate(routes):
            if len(route) <= 4:
                continue
            old_dist = route_dists[idx]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j == i+1:
                        continue
                    new_route = apply_2opt(route, i, j)
                    new_dist = route_distance(new_route)
                    new_max = max(route_dists[:idx] + [new_dist] + route_dists[idx+1:])
                    if new_max < best_new_max or (new_max == best_new_max and (best_move is None or idx < best_move[3] or (idx == best_move[3] and i < best_move[0]))):
                        best_new_max = new_max
                        best_move = (i, j, new_route, idx, new_dist)
        if best_move is not None and best_new_max < best_max:
            _, _, new_route, idx, new_dist = best_move
            routes[idx] = new_route
            route_dists[idx] = new_dist
            improved = True
            best_routes = copy_routes(routes)
            best_max = best_new_max
            report_best_vrp(best_routes)
            stagnation = 0
            continue
        
        # Best improvement relocate (inter-route)
        best_move = None
        best_new_max = best_max
        for i in range(truck_count):
            for j in range(truck_count):
                if i == j:
                    continue
                route_i = routes[i]
                route_j = routes[j]
                if len(route_i) <= 3:
                    continue
                old_i_dist = route_dists[i]
                old_j_dist = route_dists[j]
                for pos_i in range(1, len(route_i)-1):
                    cust = route_i[pos_i]
                    # Remove cust from i
                    new_i = route_i[:pos_i] + route_i[pos_i+1:]
                    new_i_dist = route_distance(new_i)
                    # Evaluate best insertion into j
                    best_pos_j = -1
                    best_new_j_dist = float('inf')
                    for pos_j in range(1, len(route_j)):
                        prev = route_j[pos_j-1]
                        nxt = route_j[pos_j]
                        inc = -distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
                        new_j_dist = old_j_dist + inc
                        if new_j_dist < best_new_j_dist or (new_j_dist == best_new_j_dist and pos_j < best_pos_j):
                            best_new_j_dist = new_j_dist
                            best_pos_j = pos_j
                    if best_pos_j == -1:
                        continue
                    new_j = route_j[:best_pos_j] + [cust] + route_j[best_pos_j:]
                    new_max = max(route_dists[:i] + [new_i_dist] + route_dists[i+1:j] + [best_new_j_dist] + route_dists[j+1:])
                    if new_max < best_new_max or (new_max == best_new_max and (best_move is None or i < best_move[0] or (i == best_move[0] and j < best_move[1]) or (i == best_move[0] and j == best_move[1] and cust < best_move[3]))):
                        best_new_max = new_max
                        best_move = (i, j, cust, new_i, new_i_dist, new_j, best_new_j_dist)
        if best_move is not None and best_new_max < best_max:
            i, j, _, new_i, new_i_dist, new_j, new_j_dist = best_move
            routes[i] = new_i
            routes[j] = new_j
            route_dists[i] = new_i_dist
            route_dists[j] = new_j_dist
            improved = True
            best_routes = copy_routes(routes)
            best_max = best_new_max
            report_best_vrp(best_routes)
            stagnation = 0
            continue
        
        # Best improvement exchange (inter-route)
        best_move = None
        best_new_max = best_max
        for i in range(truck_count):
            for j in range(i+1, truck_count):
                route_i = routes[i]
                route_j = routes[j]
                if len(route_i) <= 3 or len(route_j) <= 3:
                    continue
                old_i_dist = route_dists[i]
                old_j_dist = route_dists[j]
                for pos_i in range(1, len(route_i)-1):
                    cust_i = route_i[pos_i]
                    for pos_j in range(1, len(route_j)-1):
                        cust_j = route_j[pos_j]
                        # Remove both
                        new_i = route_i[:pos_i] + route_i[pos_i+1:]
                        new_j = route_j[:pos_j] + route_j[pos_j+1:]
                        # Insert cust_i into new_j at best position
                        best_pos_j = -1
                        best_new_j_dist = float('inf')
                        for p in range(1, len(new_j)):
                            prev = new_j[p-1]
                            nxt = new_j[p]
                            inc = -distance_matrix[prev][nxt] + distance_matrix[prev][cust_i] + distance_matrix[cust_i][nxt]
                            cand_j_dist = old_j_dist - distance_matrix[route_j[pos_j-1]][route_j[pos_j]] - distance_matrix[route_j[pos_j]][route_j[pos_j+1]] + inc  # approximate, recompute exactly
                            cand_j_dist = route_distance(new_j[:p] + [cust_i] + new_j[p:])
                            if cand_j_dist < best_new_j_dist or (cand_j_dist == best_new_j_dist and p < best_pos_j):
                                best_new_j_dist = cand_j_dist
                                best_pos_j = p
                        if best_pos_j == -1:
                            continue
                        final_j = new_j[:best_pos_j] + [cust_i] + new_j[best_pos_j:]
                        # Insert cust_j into new_i at best position
                        best_pos_i = -1
                        best_new_i_dist = float('inf')
                        for p in range(1, len(new_i)):
                            prev = new_i[p-1]
                            nxt = new_i[p]
                            inc = -distance_matrix[prev][nxt] + distance_matrix[prev][cust_j] + distance_matrix[cust_j][nxt]
                            cand_i_dist = route_distance(new_i[:p] + [cust_j] + new_i[p:])
                            if cand_i_dist < best_new_i_dist or (cand_i_dist == best_new_i_dist and p < best_pos_i):
                                best_new_i_dist = cand_i_dist
                                best_pos_i = p
                        if best_pos_i == -1:
                            continue
                        final_i = new_i[:best_pos_i] + [cust_j] + new_i[best_pos_i:]
                        new_max = max(route_dists[:i] + [best_new_i_dist] + route_dists[i+1:j] + [best_new_j_dist] + route_dists[j+1:])
                        if new_max < best_new_max or (new_max == best_new_max and (best_move is None or i < best_move[0] or (i == best_move[0] and j < best_move[1]) or (i == best_move[0] and j == best_move[1] and cust_i < best_move[3]))):
                            best_new_max = new_max
                            best_move = (i, j, final_i, best_new_i_dist, final_j, best_new_j_dist)
        if best_move is not None and best_new_max < best_max:
            i, j, final_i, dist_i, final_j, dist_j = best_move
            routes[i] = final_i
            routes[j] = final_j
            route_dists[i] = dist_i
            route_dists[j] = dist_j
            improved = True
            best_routes = copy_routes(routes)
            best_max = best_new_max
            report_best_vrp(best_routes)
            stagnation = 0
            continue
        
        if not improved:
            stagnation += 1
            if stagnation >= 5 and restart_count < max_restarts:
                # Perturbation: swap a random customer between two random routes
                if truck_count >= 2:
                    # Pick two distinct routes
                    idx1, idx2 = random.sample(range(truck_count), 2)
                    route1 = routes[idx1]
                    route2 = routes[idx2]
                    if len(route1) > 3 and len(route2) > 3:
                        # Remove a random customer from each
                        pos1 = random.randint(1, len(route1)-2)
                        pos2 = random.randint(1, len(route2)-2)
                        cust1 = route1[pos1]
                        cust2 = route2[pos2]
                        # Remove
                        new_route1 = route1[:pos1] + route1[pos1+1:]
                        new_route2 = route2[:pos2] + route2[pos2+1:]
                        # Insert cust1 into new_route2 at best position (greedy minimal increase on that route)
                        best_pos = 1
                        best_inc = float('inf')
                        for p in range(1, len(new_route2)):
                            prev = new_route2[p-1]
                            nxt = new_route2[p]
                            inc = distance_matrix[prev][cust1] + distance_matrix[cust1][nxt] - distance_matrix[prev][nxt]
                            if inc < best_inc:
                                best_inc = inc
                                best_pos = p
                        new_route2_insert = new_route2[:best_pos] + [cust1] + new_route2[best_pos:]
                        # Insert cust2 into new_route1 at best position
                        best_pos2 = 1
                        best_inc2 = float('inf')
                        for p in range(1, len(new_route1)):
                            prev = new_route1[p-1]
                            nxt = new_route1[p]
                            inc = distance_matrix[prev][cust2] + distance_matrix[cust2][nxt] - distance_matrix[prev][nxt]
                            if inc < best_inc2:
                                best_inc2 = inc
                                best_pos2 = p
                        new_route1_insert = new_route1[:best_pos2] + [cust2] + new_route1[best_pos2:]
                        # Update routes
                        routes[idx1] = new_route1_insert
                        routes[idx2] = new_route2_insert
                        route_dists = compute_route_dists(routes)
                        current_max = max(route_dists)
                        if current_max < best_max:
                            best_routes = copy_routes(routes)
                            best_max = current_max
                            report_best_vrp(best_routes)
                restart_count += 1
                stagnation = 0
        else:
            stagnation = 0
    
    return best_routes[:truck_count]