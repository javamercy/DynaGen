import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        routes += [[0, 0]] * (truck_count - (n - 1))
        return routes
    
    # Seed selection: farthest from depot and from each other
    seeds = []
    # first seed: farthest from depot
    first = max(customers, key=lambda c: distance_matrix[0][c])
    seeds.append(first)
    while len(seeds) < truck_count:
        best_cust = max(
            (c for c in customers if c not in seeds),
            key=lambda c: min(distance_matrix[c][s] for s in seeds)
        )
        seeds.append(best_cust)
    
    # Initialize routes with seeds
    routes = [[0, s, 0] for s in seeds]
    route_dist = [distance_matrix[0][s] + distance_matrix[s][0] for s in seeds]
    assigned = set(seeds)
    
    # Insertion: remaining customers sorted by distance from depot descending
    remaining = [c for c in customers if c not in assigned]
    remaining.sort(key=lambda c: distance_matrix[0][c], reverse=True)
    
    for cust in remaining:
        best_new_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                new_route_dist = route_dist[idx] - distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
                new_max = max(route_dist[:idx] + [new_route_dist] + route_dist[idx+1:])
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_route_idx = idx
                    best_pos = pos
        # Insert
        route = routes[best_route_idx]
        pos = best_pos
        prev = route[pos-1]
        nxt = route[pos]
        route_dist[best_route_idx] += -distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
        route.insert(pos, cust)
        assigned.add(cust)
    
    # Helper: compute route distance
    def route_dist_func(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    def evaluate(routes):
        return max(route_dist_func(r) for r in routes)
    
    best_routes = [r[:] for r in routes]
    best_max = evaluate(best_routes)
    
    max_iter = max(50, n * 2)
    for iteration in range(max_iter):
        improved = False
        # Intra-route 2-opt
        for idx, route in enumerate(routes):
            if len(route) <= 4:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j == i+1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist_func(new_route)
                    if new_dist < route_dist[idx]:
                        routes[idx] = new_route
                        route_dist[idx] = new_dist
                        improved = True
                        if evaluate(routes) < best_max:
                            best_routes = [r[:] for r in routes]
                            best_max = evaluate(best_routes)
                            # report_best_vrp(best_routes)
                        break
                if improved:
                    break
        if improved:
            continue
        # Inter-route relocate
        for i in range(truck_count):
            if len(routes[i]) <= 3:
                continue
            for pos_cust in range(1, len(routes[i])-1):
                cust = routes[i][pos_cust]
                for j in range(truck_count):
                    if i == j:
                        continue
                    for pos in range(1, len(routes[j])):
                        # Remove cust from route i
                        new_route_i = routes[i][:pos_cust] + routes[i][pos_cust+1:]
                        # Insert into route j at pos
                        prev_j = routes[j][pos-1]
                        nxt_j = routes[j][pos]
                        new_dist_j = route_dist_func(routes[j]) - distance_matrix[prev_j][nxt_j] + distance_matrix[prev_j][cust] + distance_matrix[cust][nxt_j]
                        # Compute new max
                        new_dists = [route_dist_func(r) for r in routes]
                        new_dists[i] = route_dist_func(new_route_i)
                        new_dists[j] = new_dist_j
                        new_max = max(new_dists)
                        if new_max < best_max:
                            # Apply
                            routes[i] = new_route_i
                            routes[j] = routes[j][:pos] + [cust] + routes[j][pos:]
                            route_dist[i] = route_dist_func(routes[i])
                            route_dist[j] = new_dist_j
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            # report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if improved:
            continue
        # Inter-route exchange (swap customers)
        for i in range(truck_count):
            if len(routes[i]) <= 3:
                continue
            for pos_i in range(1, len(routes[i])-1):
                cust_i = routes[i][pos_i]
                for j in range(i+1, truck_count):
                    if len(routes[j]) <= 3:
                        continue
                    for pos_j in range(1, len(routes[j])-1):
                        cust_j = routes[j][pos_j]
                        # Swap
                        new_route_i = routes[i][:pos_i] + [cust_j] + routes[i][pos_i+1:]
                        new_route_j = routes[j][:pos_j] + [cust_i] + routes[j][pos_j+1:]
                        new_dists = [route_dist_func(r) for r in routes]
                        new_dists[i] = route_dist_func(new_route_i)
                        new_dists[j] = route_dist_func(new_route_j)
                        new_max = max(new_dists)
                        if new_max < best_max:
                            routes[i] = new_route_i
                            routes[j] = new_route_j
                            route_dist[i] = new_dists[i]
                            route_dist[j] = new_dists[j]
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            # report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    # Ensure exactly truck_count routes
    final_routes = best_routes[:]
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes