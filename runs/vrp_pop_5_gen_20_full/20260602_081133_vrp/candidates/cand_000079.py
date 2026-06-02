import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0] * truck_count
    
    def route_length(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def insert_customer(customer, routes, route_distances):
        best_new_max = float('inf')
        best_new_route_d = float('inf')
        best_route_idx = None
        best_pos = None
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                old_d = route_distances[r_idx]
                removed = distance_matrix[route[pos-1], route[pos]]
                added = distance_matrix[route[pos-1], customer] + distance_matrix[customer, route[pos]]
                new_d = old_d - removed + added
                other_max = max(route_distances[j] for j in range(truck_count) if j != r_idx) if truck_count > 1 else 0.0
                new_max = max(new_d, other_max)
                if new_max < best_new_max or (new_max == best_new_max and new_d < best_new_route_d):
                    best_new_max = new_max
                    best_new_route_d = new_d
                    best_route_idx = r_idx
                    best_pos = pos
        routes[best_route_idx].insert(best_pos, customer)
        route_distances[best_route_idx] = best_new_route_d
    
    # Construction phase: insert customers in descending order of distance from depot
    customers = list(range(1, n))
    dist_from_depot = [distance_matrix[0, i] for i in customers]
    order = sorted(customers, key=lambda x: -dist_from_depot[x-1])
    for customer in order:
        insert_customer(customer, routes, route_distances)
    report_best_vrp(routes)
    
    # Local search: relocate each customer to best position (deterministic)
    num_scans = max(2, min(10, n // 20))
    for _ in range(num_scans):
        improved = False
        for customer in range(1, n):
            for r_idx in range(truck_count):
                if customer in routes[r_idx]:
                    current_route = routes[r_idx]
                    current_pos = current_route.index(customer)
                    current_route.pop(current_pos)
                    route_distances[r_idx] = route_length(current_route)
                    best_new_max = float('inf')
                    best_new_route_d = float('inf')
                    best_route_idx = None
                    best_pos = None
                    for r_new in range(truck_count):
                        route = routes[r_new]
                        for pos in range(1, len(route)):
                            old_d = route_distances[r_new]
                            removed = distance_matrix[route[pos-1], route[pos]]
                            added = distance_matrix[route[pos-1], customer] + distance_matrix[customer, route[pos]]
                            new_d = old_d - removed + added
                            other_max = max(route_distances[j] for j in range(truck_count) if j != r_new) if truck_count > 1 else 0.0
                            new_max = max(new_d, other_max)
                            if new_max < best_new_max or (new_max == best_new_max and new_d < best_new_route_d):
                                best_new_max = new_max
                                best_new_route_d = new_d
                                best_route_idx = r_new
                                best_pos = pos
                    routes[best_route_idx].insert(best_pos, customer)
                    route_distances[best_route_idx] = best_new_route_d
                    if best_route_idx != r_idx or best_pos != current_pos:
                        improved = True
                    break
        report_best_vrp(routes)
        if not improved:
            break
    return routes