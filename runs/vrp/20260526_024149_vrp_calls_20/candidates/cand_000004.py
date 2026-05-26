import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    
    def route_distance(route):
        if len(route) == 2:
            return distance_matrix[route[0]][route[-1]] * 2  # Actually for [0,0] it's 0? Wait, distance from 0 to 0 is 0 so total 0.
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    
    # Global best
    best_routes = [list(r) for r in routes]
    best_max = float('inf')
    
    # Construction: insert customers in order of index
    for cust in sorted(customers):
        best_new_max = float('inf')
        best_r_idx = -1
        best_pos = -1
        current_max = max(route_distance(r) for r in routes)
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                # simulate insertion
                new_route_len = route_distance(route) + distance_matrix[route[pos-1]][cust] + distance_matrix[cust][route[pos]] - distance_matrix[route[pos-1]][route[pos]]
                new_max = max(current_max, new_route_len)
                # tie-breaking by route index then pos (but pos not needed for tie-break uniquely, so we use smaller r_idx)
                if new_max < best_new_max or (new_max == best_new_max and r_idx < best_r_idx):
                    best_new_max = new_max
                    best_r_idx = r_idx
                    best_pos = pos
                    best_routes_temp = [list(r) for r in routes]
                    best_routes_temp[best_r_idx].insert(best_pos, cust)
                    # update best if this is the first found
        # Apply insertion
        routes[best_r_idx].insert(best_pos, cust)
        # Update global best
        cur_max = max(route_distance(r) for r in routes)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = [list(r) for r in routes]
            from report_best_vrp import report_best_vrp
            report_best_vrp(best_routes)
    
    # Improvement
    max_iter = 10 * n
    for iteration in range(max_iter):
        improved = False
        # Relocate moves
        for cust in sorted(customers):
            # Find current route and position
            for r_idx, route in enumerate(routes):
                if cust in route:
                    break
            # Remove cust temporarily
            route_idx = r_idx
            route = routes[route_idx]
            pos = route.index(cust)
            # Compute savings for removal
            prev = route[pos-1]
            nxt = route[pos+1]
            savings = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
            old_route_len = route_distance(route)
            new_route_len = old_route_len - savings
            # Try all insertions
            best_move = None
            best_new_max = None
            for r2_idx in range(truck_count):
                route2 = routes[r2_idx]
                for pos2 in range(1, len(route2)):
                    # compute new max
                    # Simulate insertion
                    new_len2 = route_distance(route2) + distance_matrix[route2[pos2-1]][cust] + distance_matrix[cust][route2[pos2]] - distance_matrix[route2[pos2-1]][route2[pos2]]
                    # new max candidate
                    new_max = max(
                        [new_route_len if i == route_idx else (new_len2 if i == r2_idx else route_distance(routes[i])) for i in range(truck_count)]
                    )
                    if best_new_max is None or new_max < best_new_max:
                        best_new_max = new_max
                        best_move = (r2_idx, pos2)
            if best_move is not None and best_new_max < best_max:
                # Apply move
                # Remove
                routes[route_idx].pop(pos)
                # Insert
                r2_idx, pos2 = best_move
                routes[r2_idx].insert(pos2, cust)
                best_max = best_new_max
                best_routes = [list(r) for r in routes]
                from report_best_vrp import report_best_vrp
                report_best_vrp(best_routes)
                improved = True
        # Exchange moves (swap two customers from different routes)
        for i in range(truck_count):
            for j in range(i+1, truck_count):
                route_i = routes[i]
                route_j = routes[j]
                for pos_i in range(1, len(route_i)-1):  # exclude depot
                    for pos_j in range(1, len(route_j)-1):
                        cust_i = route_i[pos_i]
                        cust_j = route_j[pos_j]
                        # Simulate swap
                        # Compute new routes
                        new_route_i = route_i[:]
                        new_route_j = route_j[:]
                        new_route_i[pos_i] = cust_j
                        new_route_j[pos_j] = cust_i
                        # Recompute distances
                        # For simplicity, compute full distances
                        temps = [list(r) for r in routes]
                        temps[i] = new_route_i
                        temps[j] = new_route_j
                        new_max = max(route_distance(r) for r in temps)
                        if new_max < best_max:
                            # Apply
                            routes[i] = new_route_i
                            routes[j] = new_route_j
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            from report_best_vrp import report_best_vrp
                            report_best_vrp(best_routes)
                            improved = True
        if not improved:
            break
    
    # After improvement, return best found
    # Ensure all routes start and end at 0
    for r in best_routes:
        if r[0] != 0 or r[-1] != 0:
            raise ValueError("Route does not start or end at depot")
    # Ensure every customer appears exactly once
    flat = [c for r in best_routes for c in r[1:-1]]
    assert sorted(flat) == sorted(customers)
    return best_routes