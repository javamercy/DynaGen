import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)  # deterministic
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    best_routes = None
    best_max = float('inf')
    
    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]
    
    # ---- Cheapest insertion construction ----
    # Seed selection: farthest-first from depot
    seeds = []
    seed0 = max(customers, key=lambda x: distance_matrix[0, x])
    seeds.append(seed0)
    while len(seeds) < truck_count:
        best_cust = None
        best_dist = -1
        for c in customers:
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c, s] for s in seeds)
            if min_dist > best_dist or (min_dist == best_dist and (best_cust is None or c < best_cust)):
                best_dist = min_dist
                best_cust = c
        seeds.append(best_cust)
    
    # Initialize routes with seeds
    routes = [[0, s, 0] for s in seeds]
    assigned = set(seeds)
    unassigned = [c for c in customers if c not in assigned]
    
    # Insert remaining customers into best route at best position (parallel insertion)
    while unassigned:
        best_delta = float('inf')
        best_cust = None
        best_route_idx = None
        best_pos = None
        for cust in unassigned:
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if delta < best_delta or (delta == best_delta and (cust < best_cust if best_cust is not None else True)):
                        best_delta = delta
                        best_cust = cust
                        best_route_idx = r_idx
                        best_pos = pos
        # Insert
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    
    report_best_vrp(routes)
    
    # ---- Tabu Search ----
    max_iter = min(200, n * truck_count * 2)
    tabu_tenure = 7
    tabu_list = {}  # customer -> iteration until which it is tabu
    current_routes = [list(r) for r in routes]
    current_max = max(route_distance(r) for r in current_routes)
    
    for iteration in range(max_iter):
        improved = False
        best_move = None
        best_new_max = float('inf')
        # 1. Explore relocate moves
        for r1 in range(truck_count):
            route1 = current_routes[r1]
            if len(route1) <= 2:
                continue
            for cust in route1[1:-1]:
                if cust in tabu_list and tabu_list[cust] > iteration:
                    continue
                for r2 in range(truck_count):
                    if r2 == r1:
                        continue
                    route2 = current_routes[r2]
                    for pos in range(1, len(route2)):
                        new_routes = [list(r) for r in current_routes]
                        new_routes[r1].remove(cust)
                        new_routes[r2].insert(pos, cust)
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_new_max - 1e-12 or (abs(new_max - best_new_max) < 1e-12 and cust < (best_move[1] if best_move else float('inf'))):
                            # Check aspiration: if tabu but yields global best, allow
                            if cust in tabu_list and tabu_list[cust] > iteration:
                                if new_max < best_max - 1e-12:
                                    best_move = ('relocate', r1, cust, r2, pos)
                                    best_new_max = new_max
                            else:
                                best_move = ('relocate', r1, cust, r2, pos)
                                best_new_max = new_max
        # 2. Explore swap moves
        if best_move is None:
            for r1 in range(truck_count):
                route1 = current_routes[r1]
                if len(route1) <= 2:
                    continue
                for cust1 in route1[1:-1]:
                    if cust1 in tabu_list and tabu_list[cust1] > iteration:
                        continue
                    for r2 in range(r1+1, truck_count):
                        route2 = current_routes[r2]
                        if len(route2) <= 2:
                            continue
                        for cust2 in route2[1:-1]:
                            if cust2 in tabu_list and tabu_list[cust2] > iteration:
                                continue
                            new_routes = [list(r) for r in current_routes]
                            idx1 = new_routes[r1].index(cust1)
                            idx2 = new_routes[r2].index(cust2)
                            new_routes[r1][idx1], new_routes[r2][idx2] = cust2, cust1
                            new_max = max(route_distance(r) for r in new_routes)
                            if new_max < best_new_max - 1e-12 or (abs(new_max - best_new_max) < 1e-12 and (cust1 < cust2)):
                                # Aspiration
                                if (cust1 in tabu_list and tabu_list[cust1] > iteration) or (cust2 in tabu_list and tabu_list[cust2] > iteration):
                                    if new_max < best_max - 1e-12:
                                        best_move = ('swap', r1, cust1, r2, cust2)
                                        best_new_max = new_max
                                else:
                                    best_move = ('swap', r1, cust1, r2, cust2)
                                    best_new_max = new_max
        # Apply best move
        if best_move is not None and best_new_max < current_max - 1e-12:
            if best_move[0] == 'relocate':
                _, r1, cust, r2, pos = best_move
                current_routes[r1].remove(cust)
                current_routes[r2].insert(pos, cust)
                tabu_list[cust] = iteration + tabu_tenure
            else:  # swap
                _, r1, cust1, r2, cust2 = best_move
                idx1 = current_routes[r1].index(cust1)
                idx2 = current_routes[r2].index(cust2)
                current_routes[r1][idx1], current_routes[r2][idx2] = cust2, cust1
                tabu_list[cust1] = iteration + tabu_tenure
                tabu_list[cust2] = iteration + tabu_tenure
            current_max = best_new_max
            if current_max < best_max - 1e-12:
                report_best_vrp(current_routes)
            improved = True
        
        # 3. If no improving inter-route move, try intra-route 2-opt
        if not improved:
            for r_idx in range(truck_count):
                route = current_routes[r_idx]
                if len(route) <= 3:
                    continue
                best_route = list(route)
                best_dist = route_distance(route)
                found = False
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < best_dist - 1e-12:
                            best_dist = new_dist
                            best_route = new_route
                            found = True
                            break
                    if found:
                        break
                if found:
                    current_routes[r_idx] = best_route
                    new_max = max(route_distance(r) for r in current_routes)
                    if new_max < current_max - 1e-12:
                        current_max = new_max
                        if current_max < best_max - 1e-12:
                            report_best_vrp(current_routes)
                        improved = True
                    break
        
        if not improved:
            break
    
    # Ensure exactly truck_count routes
    final_routes = best_routes if best_routes is not None else current_routes
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes