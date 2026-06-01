import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    # sort customers by distance to depot descending
    customers = list(range(1, n))
    customers.sort(key=lambda c: distance_matrix[0, c], reverse=True)
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def insert_customer(route, pos, cust):
        return route[:pos] + [cust] + route[pos:]
    
    # Greedy insertion with sorted customers
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
                # tie-breaking: lower route index, then lower position index
                if new_max < best_max or (new_max == best_max and (r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                    best_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        route = routes[best_route_idx]
        routes[best_route_idx] = insert_customer(route, best_pos, cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)
    initial_max = best_max
    
    # Adaptive tolerance: starts at 10% of initial max, decays linearly to 0 over n^2 iterations
    max_iter = n * n
    tolerance = 0.1 * initial_max
    tolerance_decay = tolerance / max_iter
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        # 2-opt for each route
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
            tolerance = max(0, tolerance - tolerance_decay)
            continue
        # Relocate from longest route
        max_dist = max_route_distance(routes)
        longest_indices = [i for i, r in enumerate(routes) if route_distance(r) == max_dist]
        for r_idx in longest_indices:
            if len(routes[r_idx]) <= 3:
                continue
            for pos in range(1, len(routes[r_idx])-1):
                cust = routes[r_idx][pos]
                for other_idx, other_route in enumerate(routes):
                    if other_idx == r_idx:
                        continue
                    for other_pos in range(1, len(other_route)):
                        new_other = insert_customer(other_route, other_pos, cust)
                        new_self = routes[r_idx][:pos] + routes[r_idx][pos+1:]
                        new_routes = list(routes)
                        new_routes[r_idx] = new_self
                        new_routes[other_idx] = new_other
                        new_max = max_route_distance(new_routes)
                        if new_max < best_max or (new_max <= best_max + tolerance and new_max < max_route_distance(routes)):
                            best_max = min(best_max, new_max)
                            best_routes = [list(r) for r in new_routes]
                            routes = new_routes
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            tolerance = max(0, tolerance - tolerance_decay)
            continue
        # Swap between routes (customer exchange)
        for i in range(truck_count):
            for j in range(i+1, truck_count):
                route_i = routes[i]
                route_j = routes[j]
                if len(route_i) <= 2 or len(route_j) <= 2:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust_i = route_i[pos_i]
                    for pos_j in range(1, len(route_j)-1):
                        cust_j = route_j[pos_j]
                        # new routes after swap
                        new_route_i = route_i[:pos_i] + [cust_j] + route_i[pos_i+1:]
                        new_route_j = route_j[:pos_j] + [cust_i] + route_j[pos_j+1:]
                        new_routes = list(routes)
                        new_routes[i] = new_route_i
                        new_routes[j] = new_route_j
                        new_max = max_route_distance(new_routes)
                        if new_max < best_max or (new_max <= best_max + tolerance and new_max < max_route_distance(routes)):
                            best_max = min(best_max, new_max)
                            best_routes = [list(r) for r in new_routes]
                            routes = new_routes
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            tolerance = max(0, tolerance - tolerance_decay)
    
    # Ensure routes start and end at depot
    for r in best_routes:
        assert r[0] == 0 and r[-1] == 0
    return best_routes