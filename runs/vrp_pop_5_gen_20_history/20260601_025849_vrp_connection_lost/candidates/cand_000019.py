import numpy as np
import math

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
    
    # Greedy insertion with balanced tie-breaking
    for cust in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        best_cur_dist = 0.0
        best_cur_len = 0
        current_max = max_route_distance(routes)
        for r_idx, route in enumerate(routes):
            cur_dist = route_distance(route)
            cur_len = len(route) - 2  # customers
            for pos in range(1, len(route)):
                new_route = insert_customer(route, pos, cust)
                new_route_dist = route_distance(new_route)
                other_routes = [routes[i] for i in range(truck_count) if i != r_idx]
                other_max = max(route_distance(r) for r in other_routes) if other_routes else 0.0
                new_max = max(new_route_dist, other_max)
                # Primary key: new_max; secondary keys: cur_dist, cur_len, r_idx, pos
                if (new_max < best_max or
                    (new_max == best_max and (cur_dist < best_cur_dist or
                     (cur_dist == best_cur_dist and (cur_len < best_cur_len or
                      (cur_len == best_cur_len and (r_idx < best_route_idx or
                       (r_idx == best_route_idx and pos < best_pos))))))):
                    best_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
                    best_cur_dist = cur_dist
                    best_cur_len = cur_len
        # Insert at best position
        route = routes[best_route_idx]
        routes[best_route_idx] = insert_customer(route, best_pos, cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)
    report_best_vrp(best_routes)
    
    # Local search improvement
    improved = True
    max_iter = n * n  # bounded
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
                            report_best_vrp(best_routes)
                        break
                if improved:
                    break
        if improved:
            continue
        # Relocate: try moving a customer from the longest route to another
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
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [list(r) for r in new_routes]
                            routes = new_routes
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_routes