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
        current_max = max_route_distance(routes)
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = insert_customer(route, pos, cust)
                new_route_dist = route_distance(new_route)
                other_routes = [routes[i] for i in range(truck_count) if i != r_idx]
                other_max = max(route_distance(r) for r in other_routes) if other_routes else 0.0
                new_max = max(new_route_dist, other_max)
                if new_max < best_max or (new_max == best_max and (r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                    best_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        route = routes[best_route_idx]
        routes[best_route_idx] = insert_customer(route, best_pos, cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)
    report_best_vrp(best_routes)
    
    # Or-opt improvement (replaces 2-opt and relocate)
    max_iter = n * n
    iter_count = 0
    improved = True
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        # Intra-route Or-opt: move segment of size 1..3 within same route
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for seg_len in [3, 2, 1]:
                if improved:
                    break
                for start in range(1, len(route)-seg_len):
                    if improved:
                        break
                    segment = route[start:start+seg_len]
                    # Remove segment (endpoints are not depot)
                    remaining = route[:start] + route[start+seg_len:]
                    # Try inserting segment at every other position (excluding original)
                    for pos in range(1, len(remaining)):
                        if pos == start:  # skip original position (approximately)
                            continue
                        new_route = remaining[:pos] + segment + remaining[pos:]
                        if route_distance(new_route) < route_distance(route):
                            routes[r_idx] = new_route
                            improved = True
                            current_max = max_route_distance(routes)
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                            break
        if improved:
            continue
        # Inter-route Or-opt: move segment from one route to another
        for r_idx, route in enumerate(routes):
            if improved:
                break
            if len(route) <= 3:
                continue
            for seg_len in [3, 2, 1]:
                if improved:
                    break
                for start in range(1, len(route)-seg_len):
                    if improved:
                        break
                    segment = route[start:start+seg_len]
                    # Remove segment from source
                    source_remaining = route[:start] + route[start+seg_len:]
                    # Try inserting into each other route
                    for other_idx, other_route in enumerate(routes):
                        if other_idx == r_idx:
                            continue
                        for pos in range(1, len(other_route)):
                            new_other = other_route[:pos] + segment + other_route[pos:]
                            new_routes = list(routes)
                            new_routes[r_idx] = source_remaining
                            new_routes[other_idx] = new_other
                            new_max = max_route_distance(new_routes)
                            if new_max < best_max:
                                best_max = new_max
                                best_routes = [list(r) for r in new_routes]
                                routes = new_routes
                                improved = True
                                report_best_vrp(best_routes)
                                break
                        if improved:
                            break
    return best_routes