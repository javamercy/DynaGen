import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def insert_customer(route, pos, cust):
        return route[:pos] + [cust] + route[pos:]
    
    # Greedy insertion (same as parent)
    for cust in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = insert_customer(route, pos, cust)
                new_route_dist = route_distance(new_route)
                other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx) if truck_count > 1 else 0.0
                new_max = max(new_route_dist, other_max)
                if new_max < best_max or (new_max == best_max and (r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                    best_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        route = routes[best_route_idx]
        routes[best_route_idx] = insert_customer(route, best_pos, cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)
    
    # Refinement: targeted longest route reduction and 2-opt
    max_iter = n * n
    for _ in range(max_iter):
        improved = False
        current_max = max_route_distance(routes)
        longest_indices = [i for i, r in enumerate(routes) if route_distance(r) == current_max]
        if not longest_indices:
            break
        r_idx = longest_indices[0]  # First longest route (deterministic)
        route = routes[r_idx]
        # Try to relocate customers from longest route to others
        for pos in range(1, len(route)-1):
            cust = route[pos]
            for other_idx, other_route in enumerate(routes):
                if other_idx == r_idx:
                    continue
                for other_pos in range(1, len(other_route)):
                    new_other = insert_customer(other_route, other_pos, cust)
                    new_self = route[:pos] + route[pos+1:]
                    new_routes = [list(r) for r in routes]
                    new_routes[r_idx] = new_self
                    new_routes[other_idx] = new_other
                    new_max = max_route_distance(new_routes)
                    if new_max < best_max:
                        best_max = new_max
                        best_routes = [list(r) for r in new_routes]
                        routes = new_routes
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Try inter-route swaps between longest route and others
        for pos1 in range(1, len(routes[r_idx])-1):
            cust1 = routes[r_idx][pos1]
            for other_idx, other_route in enumerate(routes):
                if other_idx == r_idx:
                    continue
                for pos2 in range(1, len(other_route)-1):
                    cust2 = other_route[pos2]
                    # Swap cust1 and cust2
                    new_route1 = routes[r_idx][:pos1] + [cust2] + routes[r_idx][pos1+1:]
                    new_route2 = other_route[:pos2] + [cust1] + other_route[pos2+1:]
                    new_routes = [list(r) for r in routes]
                    new_routes[r_idx] = new_route1
                    new_routes[other_idx] = new_route2
                    new_max = max_route_distance(new_routes)
                    if new_max < best_max:
                        best_max = new_max
                        best_routes = [list(r) for r in new_routes]
                        routes = new_routes
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 2-opt on each route
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        routes[r_idx] = new_route
                        improved = True
                        current_max = max_route_distance(routes)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
        # Call report_best_vrp when improved
        if best_max < max_route_distance(routes):
            # report best
            pass
    # Final check: preserve best found
    return best_routes