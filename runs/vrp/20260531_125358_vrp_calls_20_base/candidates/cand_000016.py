import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    
    # If truck_count >= n, each customer on its own route
    if truck_count >= n:
        routes = []
        for i in range(truck_count):
            if i < len(customers):
                routes.append([0, customers[i], 0])
            else:
                routes.append([0,0])
        return routes
    
    # Helper: compute total distance of a route
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    # Helper: find best insertion position and cost increase for a customer in a route
    def best_insertion(customer, route):
        best_pos = -1
        best_inc = float('inf')
        for i in range(1, len(route)):
            prev = route[i-1]
            next_ = route[i]
            inc = distance_matrix[prev, customer] + distance_matrix[customer, next_] - distance_matrix[prev, next_]
            if inc < best_inc:
                best_inc = inc
                best_pos = i
        return best_pos, best_inc
    
    # Regret-2 construction
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0 for _ in range(truck_count)]
    remaining = set(customers)
    
    while remaining:
        regret_list = []
        for c in remaining:
            incs = []
            for r_idx, route in enumerate(routes):
                pos, inc = best_insertion(c, route)
                incs.append((inc, pos, r_idx))
            incs.sort(key=lambda x: (x[0], x[2]))  # deterministic tie by route index
            if len(incs) >= 2:
                best_inc = incs[0][0]
                second_best_inc = incs[1][0]
                regret = second_best_inc - best_inc
            else:
                best_inc = incs[0][0]
                regret = 0.0
            best_pos = incs[0][1]
            best_route = incs[0][2]
            # For tie-breaking among customers: prefer larger regret, then smaller best_inc, then larger customer id (to be deterministic)
            regret_list.append(( -regret, best_inc, c, best_pos, best_route ))
        # Sort by regret descending, then best_inc ascending, then customer id descending
        regret_list.sort(key=lambda x: (x[0], x[1], -x[2]))
        _, best_inc, customer, best_pos, best_route = regret_list[0]
        routes[best_route].insert(best_pos, customer)
        route_dists[best_route] = route_distance(routes[best_route])
        remaining.remove(customer)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp([list(r) for r in routes])
    
    # Intensified local search (best-improvement)
    def evaluate_move(new_route_i, new_route_j, i, j):
        new_dists = route_dists[:]
        new_dists[i] = route_distance(new_route_i)
        if j != i:
            new_dists[j] = route_distance(new_route_j)
        return max(new_dists)
    
    improved = True
    iteration = 0
    max_iter = n * truck_count * 2
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        best_new_max = best_max
        best_move = None
        
        # Intra-route 2-opt
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-1):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < route_dists[r_idx]:
                        new_max = max(new_dist, max(route_dists[:r_idx] + route_dists[r_idx+1:]))
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('2opt', r_idx, i, j)
        
        # Inter-route relocate
        for src_idx, src_route in enumerate(routes):
            if len(src_route) <= 2:
                continue
            for pos in range(1, len(src_route)-1):
                customer = src_route[pos]
                new_src_route = src_route[:pos] + src_route[pos+1:]
                for dst_idx, dst_route in enumerate(routes):
                    if dst_idx == src_idx:
                        continue
                    for ins_pos in range(1, len(dst_route)):
                        new_dst_route = dst_route[:ins_pos] + [customer] + dst_route[ins_pos:]
                        new_max = evaluate_move(new_src_route, new_dst_route, src_idx, dst_idx)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('relocate', src_idx, pos, dst_idx, ins_pos)
        
        # Inter-route swap
        for i_idx, i_route in enumerate(routes):
            if len(i_route) <= 2:
                continue
            for i_pos in range(1, len(i_route)-1):
                cust_i = i_route[i_pos]
                for j_idx in range(i_idx+1, len(routes)):
                    j_route = routes[j_idx]
                    if len(j_route) <= 2:
                        continue
                    for j_pos in range(1, len(j_route)-1):
                        cust_j = j_route[j_pos]
                        new_i_route = i_route[:i_pos] + [cust_j] + i_route[i_pos+1:]
                        new_j_route = j_route[:j_pos] + [cust_i] + j_route[j_pos+1:]
                        new_max = evaluate_move(new_i_route, new_j_route, i_idx, j_idx)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('swap', i_idx, i_pos, j_idx, j_pos)
        
        # Inter-route or-opt: move a segment of length 1 or 2
        for src_idx, src_route in enumerate(routes):
            if len(src_route) <= 3:
                continue
            for start in range(1, len(src_route)-2):
                for length in [1, 2]:
                    end = start + length - 1
                    if end >= len(src_route)-1:
                        continue
                    segment = src_route[start:end+1]
                    new_src_route = src_route[:start] + src_route[end+1:]
                    for dst_idx, dst_route in enumerate(routes):
                        if dst_idx == src_idx:
                            continue
                        for ins_pos in range(1, len(dst_route)):
                            new_dst_route = dst_route[:ins_pos] + segment + dst_route[ins_pos:]
                            new_max = evaluate_move(new_src_route, new_dst_route, src_idx, dst_idx)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('oropt', src_idx, start, end, dst_idx, ins_pos)
        
        # Apply best move if improvement
        if best_move is not None and best_new_max < best_max:
            move_type = best_move[0]
            if move_type == '2opt':
                r_idx, i, j = best_move[1], best_move[2], best_move[3]
                routes[r_idx] = routes[r_idx][:i] + routes[r_idx][i:j+1][::-1] + routes[r_idx][j+1:]
            elif move_type == 'relocate':
                src_idx, pos, dst_idx, ins_pos = best_move[1], best_move[2], best_move[3], best_move[4]
                customer = routes[src_idx].pop(pos)
                routes[dst_idx].insert(ins_pos, customer)
            elif move_type == 'swap':
                i_idx, i_pos, j_idx, j_pos = best_move[1], best_move[2], best_move[3], best_move[4]
                routes[i_idx][i_pos], routes[j_idx][j_pos] = routes[j_idx][j_pos], routes[i_idx][i_pos]
            elif move_type == 'oropt':
                src_idx, start, end, dst_idx, ins_pos = best_move[1], best_move[2], best_move[3], best_move[4], best_move[5]
                segment = routes[src_idx][start:end+1]
                del routes[src_idx][start:end+1]
                routes[dst_idx][ins_pos:ins_pos] = segment
            # Update distances
            route_dists = [route_distance(r) for r in routes]
            best_max = max(route_dists)
            best_routes = [list(r) for r in routes]
            report_best_vrp([list(r) for r in routes])
            improved = True
    
    # Adaptive balancing (from longest to shortest)
    def route_length(route):
        return route_distance(route)
    
    no_improve_count = 0
    max_balance_iter = n * truck_count
    for _ in range(max_balance_iter):
        max_idx = max(range(truck_count), key=lambda i: route_dists[i])
        min_idx = min(range(truck_count), key=lambda i: route_dists[i])
        if max_idx == min_idx or route_dists[max_idx] == route_dists[min_idx]:
            break
        max_route = routes[max_idx]
        min_route = routes[min_idx]
        # Evaluate moving each customer from max_route to min_route
        best_cust = None
        best_reduction = 0
        for pos in range(1, len(max_route)-1):
            cust = max_route[pos]
            new_max_route = max_route[:pos] + max_route[pos+1:]
            new_max_len = route_length(new_max_route)
            # Find best insertion in min_route
            best_inc = float('inf')
            best_pos = -1
            for p in range(1, len(min_route)):
                inc = distance_matrix[min_route[p-1], cust] + distance_matrix[cust, min_route[p]] - distance_matrix[min_route[p-1], min_route[p]]
                if inc < best_inc:
                    best_inc = inc
                    best_pos = p
            new_min_route = min_route[:best_pos] + [cust] + min_route[best_pos:]
            new_min_len = route_length(new_min_route)
            other_lengths = [route_dists[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
            new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
            reduction = route_dists[max_idx] - new_max_global  # positive if max reduces
            if reduction > best_reduction:
                best_reduction = reduction
                best_cust = (cust, pos, best_pos)
        if best_cust is not None and best_reduction > 0:
            cust, remove_pos, insert_pos = best_cust
            # Remove from max_route
            new_max = [node for node in max_route if node != cust]
            # Insert into min_route
            new_min = min_route[:insert_pos] + [cust] + min_route[insert_pos:]
            routes[max_idx] = new_max
            routes[min_idx] = new_min
            route_dists[max_idx] = route_length(new_max)
            route_dists[min_idx] = route_length(new_min)
            best_max = max(route_dists)
            best_routes = [list(r) for r in routes]
            report_best_vrp([list(r) for r in routes])
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
    
    # Final 2-opt on each route
    for i in range(truck_count):
        if len(routes[i]) > 2:
            route = routes[i]
            improved = True
            it = 0
            max_iter_2opt = max(10, len(route) * 2)
            while improved and it < max_iter_2opt:
                improved = False
                it += 1
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        if route_length(new_route) < route_length(route):
                            route = new_route
                            improved = True
            routes[i] = route
            route_dists[i] = route_length(route)
    best_max = max(route_dists)
    best_routes = [list(r) for r in routes]
    report_best_vrp([list(r) for r in routes])
    
    return best_routes