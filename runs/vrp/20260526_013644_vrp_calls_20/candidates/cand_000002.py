import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # sort customers by distance from depot descending
    customers.sort(key=lambda c: distance_matrix[0, c], reverse=True)
    
    # initialize routes: each route is a list of nodes, starting and ending at 0
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count
    
    best_max = float('inf')
    best_routes = None
    
    def update_best():
        nonlocal best_max, best_routes
        max_len = max(route_lengths)
        if max_len < best_max:
            best_max = max_len
            best_routes = [r[:] for r in routes]
    
    # helper: best insertion position in a route
    def best_insertion(route, route_len, customer):
        best_pos = -1
        best_increase = float('inf')
        for i in range(1, len(route)):
            prev = route[i-1]
            next = route[i]
            increase = distance_matrix[prev, customer] + distance_matrix[customer, next] - distance_matrix[prev, next]
            if increase < best_increase:
                best_increase = increase
                best_pos = i
        return best_pos, best_increase
    
    # insert all customers
    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_new_max = float('inf')
        for r_idx in range(truck_count):
            route = routes[r_idx]
            route_len = route_lengths[r_idx]
            pos, inc = best_insertion(route, route_len, cust)
            new_len = route_len + inc
            # compute new max across all routes
            # we need max of current lengths with new_len replacing old for this route
            # compute efficiently: max among other routes and new_len
            other_max = max(route_lengths[:r_idx] + route_lengths[r_idx+1:]) if truck_count > 1 else 0
            new_max = max(other_max, new_len)
            if new_max < best_new_max or (new_max == best_new_max and r_idx < best_route_idx):
                best_new_max = new_max
                best_route_idx = r_idx
                best_pos = pos
        # insert into best route
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        # update length: recalc from scratch to avoid floating errors
        old_len = route_lengths[best_route_idx]
        new_len = 0.0
        for i in range(len(route)-1):
            new_len += distance_matrix[route[i], route[i+1]]
        route_lengths[best_route_idx] = new_len
    
    update_best()
    
    # Improvement: 2-opt on each route
    for r_idx in range(truck_count):
        route = routes[r_idx]
        improved = True
        max_iter = len(route)  # bounded
        for _ in range(max_iter):
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    # check if reversing segment i..j reduces distance
                    # current contribution: dist(route[i-1], route[i]) + dist(route[j], route[j+1])
                    # new: dist(route[i-1], route[j]) + dist(route[i], route[j+1])
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new < old - 1e-9:
                        # reverse segment
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
            if not improved:
                break
        # recalc length
        new_len = 0.0
        for i in range(len(route)-1):
            new_len += distance_matrix[route[i], route[i+1]]
        route_lengths[r_idx] = new_len
    
    update_best()
    
    return best_routes