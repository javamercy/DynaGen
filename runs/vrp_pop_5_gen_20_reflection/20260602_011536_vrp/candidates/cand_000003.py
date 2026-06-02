import numpy as np
from copy import deepcopy


def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Initialize empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    unvisited = set(range(1, n))
    
    # Regret-insertion construction
    while unvisited:
        best_customer = None
        best_regret = -1
        best_route_idx = None
        best_pos = None
        best_incr = None
        
        for cust in unvisited:
            incr_list = []
            for r_idx, route in enumerate(routes):
                # Find best insertion position in this route
                best_incr_route = float('inf')
                best_pos_route = -1
                for i in range(1, len(route)):
                    # Insert between i-1 and i
                    delta = (distance_matrix[route[i-1], cust] +
                             distance_matrix[cust, route[i]] -
                             distance_matrix[route[i-1], route[i]])
                    if delta < best_incr_route:
                        best_incr_route = delta
                        best_pos_route = i
                incr_list.append((best_incr_route, best_pos_route, r_idx))
            # Sort by increment to get best and second best
            incr_list.sort(key=lambda x: (x[0], x[2]))  # tie by route index
            best_incr_cust = incr_list[0][0]
            second_best_incr = incr_list[1][0] if len(incr_list) > 1 else float('inf')
            regret = second_best_incr - best_incr_cust
            if regret > best_regret:
                best_regret = regret
                best_customer = cust
                best_route_idx = incr_list[0][2]
                best_pos = incr_list[0][1]
                best_incr = best_incr_cust
            elif regret == best_regret and cust < best_customer:
                # deterministic tie-break: smaller customer index
                best_customer = cust
                best_route_idx = incr_list[0][2]
                best_pos = incr_list[0][1]
                best_incr = best_incr_cust
        
        # Insert best customer
        route = routes[best_route_idx]
        route.insert(best_pos, best_customer)
        unvisited.remove(best_customer)
    
    # Improvement phase
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def total_distances(routes):
        return [route_distance(r) for r in routes]
    
    
    def two_opt(route):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+2, len(route)-1):
                    old = (distance_matrix[route[i-1], route[i]] +
                           distance_matrix[route[j], route[j+1]])
                    new = (distance_matrix[route[i-1], route[j]] +
                           distance_matrix[route[i], route[j+1]])
                    if new < old:
                        route[i:j+1] = route[i:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return route
    
    
    def relocate(routes, max_dist_idx):
        # Try to move a customer from the max distance route to another route to reduce max
        best_max = float('inf')
        best_move = None
        route_max = routes[max_dist_idx]
        for pos in range(1, len(route_max)-1):
            cust = route_max.pop(pos)
            for r_idx, route in enumerate(routes):
                if r_idx == max_dist_idx:
                    continue
                for ins in range(1, len(route)):
                    route.insert(ins, cust)
                    new_dist = total_distances(routes)
                    new_max = max(new_dist)
                    if new_max < best_max:
                        best_max = new_max
                        best_move = (max_dist_idx, pos, r_idx, ins, cust)
                    route.pop(ins)
            route_max.insert(pos, cust)
        if best_move:
            # Apply the move
            src_idx, src_pos, dst_idx, dst_pos, cust = best_move
            route_src = routes[src_idx]
            route_src.pop(src_pos)
            route_dst = routes[dst_idx]
            route_dst.insert(dst_pos, cust)
            return True
        return False
    
    
    def exchange(routes):
        # Try swapping two customers from different routes
        best_max = float('inf')
        best_swap = None
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                route_i = routes[i]
                route_j = routes[j]
                for pos_i in range(1, len(route_i)-1):
                    for pos_j in range(1, len(route_j)-1):
                        # swap
                        route_i[pos_i], route_j[pos_j] = route_j[pos_j], route_i[pos_i]
                        new_dist = total_distances(routes)
                        new_max = max(new_dist)
                        if new_max < best_max:
                            best_max = new_max
                            best_swap = (i, pos_i, j, pos_j)
                        # swap back
                        route_i[pos_i], route_j[pos_j] = route_j[pos_j], route_i[pos_i]
        if best_swap:
            i, pos_i, j, pos_j = best_swap
            routes[i][pos_i], routes[j][pos_j] = routes[j][pos_j], routes[i][pos_i]
            return True
        return False
    
    
    report_best_vrp(routes)
    best_routes = deepcopy(routes)
    best_max = max(total_distances(routes))
    
    # Improvement loops (bounded by instance size)
    for iteration in range(2 * n):
        # Intra-route 2-opt on all routes
        for idx in range(truck_count):
            routes[idx] = two_opt(routes[idx])
        # Compute distances
        dists = total_distances(routes)
        max_idx = np.argmax(dists)
        # Inter-route: relocate from max route
        improved = relocate(routes, max_idx)
        if not improved:
            improved = exchange(routes)
        dists = total_distances(routes)
        current_max = max(dists)
        if current_max < best_max:
            best_max = current_max
            best_routes = deepcopy(routes)
            report_best_vrp(best_routes)
        # If no improvement, break early to avoid infinite loop
        if not improved:
            break
    
    return best_routes