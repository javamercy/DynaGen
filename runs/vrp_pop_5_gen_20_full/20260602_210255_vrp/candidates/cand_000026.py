import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes
    
    # Helper functions
    def route_dist(route):
        if len(route) <= 1:
            return 0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def max_dist(routes):
        return max(route_dist(r) for r in routes)
    
    # Step 1: Farthest-point seeds
    seeds = []
    # First seed: farthest from depot
    first_seed = max(customers, key=lambda c: distance_matrix[0, c])
    seeds.append(first_seed)
    while len(seeds) < truck_count:
        best_cust = None
        best_min_dist = -1
        for c in customers:
            if c in seeds:
                continue
            min_dist_to_seeds = min(distance_matrix[c, s] for s in seeds)
            if min_dist_to_seeds > best_min_dist:
                best_min_dist = min_dist_to_seeds
                best_cust = c
        seeds.append(best_cust)
    
    routes = [[0, s, 0] for s in seeds]
    route_dists = [route_dist(r) for r in routes]
    assigned = set(seeds)
    
    # Step 2: Regret insertion
    unassigned = set(customers) - assigned
    while unassigned:
        best_regret = -float('inf')
        best_cust = None
        best_insertion = None  # (route_idx, pos, new_max)
        for cust in unassigned:
            # Evaluate insertion into each route, store best position and resulting max
            best_max_for_cust = float('inf')
            second_best_max = float('inf')
            best_route_idx = -1
            best_pos = -1
            for idx, route in enumerate(routes):
                # Find best insertion position within this route
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    next_ = route[pos]
                    new_dist = route_dists[idx] - distance_matrix[prev, next_] + distance_matrix[prev, cust] + distance_matrix[cust, next_]
                    new_max = max(route_dists[:idx] + [new_dist] + route_dists[idx+1:])
                    if new_max < best_max_for_cust:
                        second_best_max = best_max_for_cust
                        best_max_for_cust = new_max
                        best_route_idx = idx
                        best_pos = pos
                    elif new_max < second_best_max:
                        second_best_max = new_max
            # Calculate regret
            if second_best_max == float('inf'):
                regret = 0
            else:
                regret = second_best_max - best_max_for_cust
            if regret > best_regret:
                best_regret = regret
                best_cust = cust
                best_insertion = (best_route_idx, best_pos, best_max_for_cust)
        # Insert best customer
        idx, pos, new_max_for_cust = best_insertion
        route = routes[idx]
        prev = route[pos-1]
        next_ = route[pos]
        route_dists[idx] += -distance_matrix[prev, next_] + distance_matrix[prev, best_cust] + distance_matrix[best_cust, next_]
        route.insert(pos, best_cust)
        assigned.add(best_cust)
        unassigned.remove(best_cust)
    
    # Ensure exactly truck_count routes
    while len(routes) < truck_count:
        routes.append([0, 0])
        route_dists.append(0)
    
    best_routes = [r[:] for r in routes]
    best_max = max_dist(routes)
    report_best_vrp(best_routes)
    
    # Step 3: Best-improvement local search with restarts
    max_restarts = 5
    for restart in range(max_restarts):
        local_improved = True
        max_iter = n * n  # bounded
        iteration = 0
        while local_improved and iteration < max_iter:
            local_improved = False
            current_max = max_dist(routes)
            best_move = None
            best_improvement = 0
            # 2-opt moves
            for idx, route in enumerate(routes):
                if len(route) <= 4:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        if j == i+1:
                            continue
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_dist(new_route)
                        new_max = max(route_dists[:idx] + [new_dist] + route_dists[idx+1:])
                        improvement = current_max - new_max
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = ('2opt', idx, new_route, new_dist)
            # Relocate moves
            for i in range(truck_count):
                if len(routes[i]) <= 3:
                    continue
                for pos_i in range(1, len(routes[i])-1):
                    cust = routes[i][pos_i]
                    # Remove from i
                    new_i = routes[i][:pos_i] + routes[i][pos_i+1:]
                    new_i_dist = route_dist(new_i)
                    for j in range(truck_count):
                        if i == j:
                            continue
                        route_j = routes[j]
                        for pos_j in range(1, len(route_j)+1):
                            new_j = route_j[:pos_j] + [cust] + route_j[pos_j:]
                            new_j_dist = route_dist(new_j)
                            new_max = max(route_dists[:i] + [new_i_dist] + route_dists[i+1:j] + [new_j_dist] + route_dists[j+1:])
                            improvement = current_max - new_max
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_move = ('relocate', i, j, new_i, new_j, new_i_dist, new_j_dist)
            # Exchange moves
            for i in range(truck_count):
                if len(routes[i]) <= 3:
                    continue
                for pos_i in range(1, len(routes[i])-1):
                    cust_i = routes[i][pos_i]
                    for j in range(i+1, truck_count):
                        if len(routes[j]) <= 3:
                            continue
                        for pos_j in range(1, len(routes[j])-1):
                            cust_j = routes[j][pos_j]
                            # Remove both from their routes
                            new_i = routes[i][:pos_i] + routes[i][pos_i+1:]
                            new_j = routes[j][:pos_j] + routes[j][pos_j+1:]
                            # Insert cust_i into best position in new_j
                            best_inc_j = float('inf')
                            best_pos_j = -1
                            for p in range(1, len(new_j)+1):
                                temp_j = new_j[:p] + [cust_i] + new_j[p:]
                                dist_j = route_dist(temp_j)
                                if dist_j < best_inc_j:
                                    best_inc_j = dist_j
                                    best_pos_j = p
                            final_j = new_j[:best_pos_j] + [cust_i] + new_j[best_pos_j:]
                            final_j_dist = route_dist(final_j)
                            # Insert cust_j into best position in new_i
                            best_inc_i = float('inf')
                            best_pos_i = -1
                            for p in range(1, len(new_i)+1):
                                temp_i = new_i[:p] + [cust_j] + new_i[p:]
                                dist_i = route_dist(temp_i)
                                if dist_i < best_inc_i:
                                    best_inc_i = dist_i
                                    best_pos_i = p
                            final_i = new_i[:best_pos_i] + [cust_j] + new_i[best_pos_i:]
                            final_i_dist = route_dist(final_i)
                            new_max = max(route_dists[:i] + [final_i_dist] + route_dists[i+1:j] + [final_j_dist] + route_dists[j+1:])
                            improvement = current_max - new_max
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_move = ('exchange', i, j, final_i, final_j, final_i_dist, final_j_dist)
            if best_move is not None:
                if best_move[0] == '2opt':
                    idx, new_route, new_dist = best_move[1], best_move[2], best_move[3]
                    routes[idx] = new_route
                    route_dists[idx] = new_dist
                elif best_move[0] == 'relocate':
                    i, j, new_i, new_j, new_i_dist, new_j_dist = best_move[1:]
                    routes[i] = new_i
                    routes[j] = new_j
                    route_dists[i] = new_i_dist
                    route_dists[j] = new_j_dist
                else:  # exchange
                    i, j, final_i, final_j, final_i_dist, final_j_dist = best_move[1:]
                    routes[i] = final_i
                    routes[j] = final_j
                    route_dists[i] = final_i_dist
                    route_dists[j] = final_j_dist
                local_improved = True
                new_max = max_dist(routes)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                current_max = new_max
            iteration += 1
        # Perturbation for next restart
        if restart < max_restarts - 1:
            num_to_perturb = max(1, int(n * 0.1))
            for _ in range(num_to_perturb):
                non_empty = [idx for idx, r in enumerate(routes) if len(r) > 2]
                if not non_empty:
                    continue
                src = random.choice(non_empty)
                if len(routes[src]) <= 3:
                    continue
                pos_src = random.randint(1, len(routes[src])-2)
                cust = routes[src].pop(pos_src)
                dst = random.randint(0, truck_count-1)
                if len(routes[dst]) == 2:
                    pos_dst = 1
                else:
                    pos_dst = random.randint(1, len(routes[dst])-1)
                routes[dst].insert(pos_dst, cust)
                route_dists[src] = route_dist(routes[src])
                route_dists[dst] = route_dist(routes[dst])
            # Reset for next restart
            current_max = max_dist(routes)
    return best_routes