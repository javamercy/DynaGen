import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    best_routes = None
    best_max_dist = float('inf')
    
    # Construction: insert customers in descending distance from depot
    cust_order = sorted(customers, key=lambda c: distance_matrix[0, c], reverse=True)
    for cust in cust_order:
        best_new_max = float('inf')
        best_r = None
        best_pos = None
        best_new_dists = None
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                # Compute new distance for this route after insertion
                new_dist = route_dists[r_idx] - distance_matrix[route[pos-1], route[pos]] \
                           + distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]]
                new_max = max(route_dists[:r_idx] + [new_dist] + route_dists[r_idx+1:])
                # Deterministic tie-break: prefer smaller new_dist when new_max equal
                if new_max < best_new_max or (new_max == best_new_max and new_dist < best_new_dists[r_idx]):
                    best_new_max = new_max
                    best_r = r_idx
                    best_pos = pos
                    best_new_dists = route_dists[:]
                    best_new_dists[r_idx] = new_dist
        # Perform insertion
        new_route = routes[best_r][:best_pos] + [cust] + routes[best_r][best_pos:]
        routes[best_r] = new_route
        route_dists[best_r] = route_length(new_route)
        # Update best solution
        if len([c for route in routes for c in route if c != 0]) == len(customers):
            max_dist = max(route_dists)
            if max_dist < best_max_dist:
                best_max_dist = max_dist
                best_routes = [route[:] for route in routes]
                # report_best_vrp assumed available
                report_best_vrp(best_routes)
    
    # Intra-route 2-opt improvement
    max_iters = n * n
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            best_improve = 0.0
            best_i = best_j = None
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    delta = -distance_matrix[route[i-1], route[i]] - distance_matrix[route[j], route[j+1]] \
                            + distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if delta < best_improve:
                        best_improve = delta
                        best_i, best_j = i, j
            if best_improve < 0:
                # Apply best 2-opt move
                new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                routes[r_idx] = new_route
                route_dists[r_idx] = route_length(new_route)
                improved = True
                new_max = max(route_dists)
                if new_max < best_max_dist:
                    best_max_dist = new_max
                    best_routes = [route[:] for route in routes]
                    report_best_vrp(best_routes)
    
    # Inter-route relocation improvement
    improved = True
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for c in customers:
            # Find current route containing c
            for r_idx, route in enumerate(routes):
                if c in route:
                    break
            old_route = route
            old_dist = route_dists[r_idx]
            new_route = [x for x in old_route if x != c]
            new_dist = route_length(new_route)
            # Try inserting into other routes
            for r2_idx, r2 in enumerate(routes):
                if r2_idx == r_idx:
                    continue
                for pos in range(1, len(r2)):
                    new_dist2 = route_dists[r2_idx] - distance_matrix[r2[pos-1], r2[pos]] \
                                + distance_matrix[r2[pos-1], c] + distance_matrix[c, r2[pos]]
                    new_max = max(route_dists[:r_idx] + [new_dist] + route_dists[r_idx+1:r2_idx] + [new_dist2] + route_dists[r2_idx+1:])
                    if new_max < best_max_dist:
                        # Apply move
                        routes[r_idx] = new_route
                        routes[r2_idx] = r2[:pos] + [c] + r2[pos:]
                        route_dists[r_idx] = new_dist
                        route_dists[r2_idx] = new_dist2
                        best_max_dist = new_max
                        best_routes = [route[:] for route in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    if best_routes is None:
        best_routes = routes
    return best_routes