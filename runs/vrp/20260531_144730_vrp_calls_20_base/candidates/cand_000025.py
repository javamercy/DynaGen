import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customer_count = n - 1
    if truck_count >= customer_count:
        routes = [[0, 0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i-1] = [0, i, 0]
        report_best_vrp(routes)
        return routes

    # Sequential cheapest insertion heuristic
    # Order customers by distance from depot (deterministic)
    order = sorted(range(1, n), key=lambda c: distance_matrix[0, c])
    
    # Initialize routes as empty
    routes = [[0, 0] for _ in range(truck_count)]
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def insert_customer(route, cust, pos):
        # pos is index in [1, len(route)-1] inclusive of end? 
        # Actually we need to insert between existing nodes, so route has indices 0..len-1
        # Insert at position pos (1 <= pos <= len(route)-1) means new route = route[:pos] + [cust] + route[pos:]
        new_route = route[:pos] + [cust] + route[pos:]
        return new_route
    
    # Insert customers one by one into best route
    for cust in order:
        best_max = float('inf')
        best_total = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx in range(truck_count):
            route = routes[r_idx]
            # possible insertion positions: after depot, before depot, but we can only insert between nodes
            # route has at least two elements (0 and 0), so positions from 1 to len(route)-1 (inclusive of before last 0? Actually before the ending 0, so len(route)-1)
            for pos in range(1, len(route)):
                new_route = insert_customer(route, cust, pos)
                # compute new max distance after insertion
                # we only changed this route, so compute its new distance and old max
                old_max = max_route_distance(routes)
                new_dist = route_distance(new_route)
                other_max = max(route_distance(routes[j]) for j in range(truck_count) if j != r_idx)
                new_max = max(new_dist, other_max)
                # tie-break: choose insertion that minimizes total distance increase
                old_total = sum(route_distance(routes[j]) for j in range(truck_count))
                new_total = old_total - route_distance(route) + new_dist
                if (new_max < best_max) or (abs(new_max - best_max) < 1e-12 and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_route_idx = r_idx
                    best_pos = pos
        # Perform insertion
        routes[best_route_idx] = insert_customer(routes[best_route_idx], cust, best_pos)
    
    # Report initial solution
    best_routes = [r[:] for r in routes]
    best_max = max_route_distance(best_routes)
    report_best_vrp(best_routes)
    
    # Intra-route 2-opt
    def two_opt(route):
        if len(route) <= 3:
            return route
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_distance(new_route) < route_distance(best) - 1e-12:
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best
    
    # Apply 2-opt to all routes initially
    for idx in range(truck_count):
        routes[idx] = two_opt(routes[idx])
    current_max = max_route_distance(routes)
    if current_max < best_max:
        best_max = current_max
        best_routes = [r[:] for r in routes]
        report_best_vrp(best_routes)
    
    # Local search
    max_iter = customer_count * truck_count
    for _ in range(max_iter):
        moved = False
        # Find longest route
        lengths = [route_distance(r) for r in routes]
        long_idx = np.argmax(lengths)
        # Try relocate from longest to shorter
        for to_route in range(truck_count):
            if to_route == long_idx or len(routes[long_idx]) <= 3:
                continue
            for cust_pos in range(1, len(routes[long_idx])-1):
                for insert_pos in range(1, len(routes[to_route])):
                    # remove customer at cust_pos from long route
                    new_long = routes[long_idx][:cust_pos] + routes[long_idx][cust_pos+1:]
                    if len(new_long) == 0:
                        new_long = [0, 0]
                    # insert into to_route
                    new_to = routes[to_route][:insert_pos] + [routes[long_idx][cust_pos]] + routes[to_route][insert_pos:]
                    # apply 2-opt to both routes immediately
                    new_long = two_opt(new_long)
                    new_to = two_opt(new_to)
                    # compute new max
                    new_max = max(route_distance(new_long), route_distance(new_to), 
                                   max(route_distance(routes[j]) for j in range(truck_count) if j not in (long_idx, to_route)))
                    if new_max < best_max - 1e-12:
                        # accept
                        routes[long_idx] = new_long
                        routes[to_route] = new_to
                        best_max = new_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                        moved = True
                        break
                if moved:
                    break
            if moved:
                break
        if moved:
            continue
        # Try exchange between longest and another route
        for r2 in range(truck_count):
            if r2 == long_idx or len(routes[long_idx]) <= 3 or len(routes[r2]) <= 3:
                continue
            for p1 in range(1, len(routes[long_idx])-1):
                for p2 in range(1, len(routes[r2])-1):
                    new_long = routes[long_idx][:]
                    new_r2 = routes[r2][:]
                    cust1 = new_long[p1]
                    cust2 = new_r2[p2]
                    new_long[p1] = cust2
                    new_r2[p2] = cust1
                    # apply 2-opt
                    new_long = two_opt(new_long)
                    new_r2 = two_opt(new_r2)
                    new_max = max(route_distance(new_long), route_distance(new_r2),
                                   max(route_distance(routes[j]) for j in range(truck_count) if j not in (long_idx, r2)))
                    if new_max < best_max - 1e-12:
                        routes[long_idx] = new_long
                        routes[r2] = new_r2
                        best_max = new_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                        moved = True
                        break
                if moved:
                    break
            if moved:
                break
        if moved:
            continue
        # If no improvement, apply 2-opt to all routes
        for idx in range(truck_count):
            routes[idx] = two_opt(routes[idx])
        current_max = max_route_distance(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
            moved = True
        if not moved:
            break
    return best_routes