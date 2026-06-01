import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = len(distance_matrix)
    customers = list(range(1, n))
    # Helper functions
    def route_distance(route):
        if len(route) <= 2:
            return 0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def compute_max_distance(routes):
        return max(route_distance(r) for r in routes)

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        maxd = compute_max_distance(routes)
        if maxd < best_max:
            best_max = maxd
            best_routes = [list(r) for r in routes]

    # Initial solution: farthest-first seed selection
    seeds = []
    # First seed: farthest from depot
    if customers:
        seed0 = max(customers, key=lambda c: distance_matrix[0, c])
        seeds.append(seed0)
    else:
        # No customers: all empty routes
        routes = [[0,0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return best_routes
    while len(seeds) < truck_count and len(seeds) < len(customers):
        best_cust = None
        best_min_dist = -1
        for c in customers:
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c, s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and c < best_cust):
                best_min_dist = min_dist
                best_cust = c
        if best_cust is not None:
            seeds.append(best_cust)
        else:
            break
    # Handle case where truck_count > len(seeds) (e.g., more trucks than customers)
    clusters = [[] for _ in range(truck_count)]
    # Assign each seed to its own cluster
    for i, s in enumerate(seeds):
        clusters[i].append(s)
    # Assign remaining customers to nearest seed
    for c in customers:
        if c in seeds:
            continue
        min_dist = float('inf')
        best_idx = 0
        for i, s in enumerate(seeds):
            d = distance_matrix[c, s]
            if d < min_dist or (d == min_dist and i < best_idx):
                min_dist = d
                best_idx = i
        clusters[best_idx].append(c)
    # Fill empty clusters with empty routes (they will be [0,0])
    # Build routes: nearest neighbor insertion from depot for each cluster
    routes = []
    for cl in clusters:
        if not cl:
            routes.append([0, 0])
            continue
        # Start at depot
        route = [0]
        unvisited = list(cl)
        current = 0
        while unvisited:
            nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        route.append(0)
        routes.append(route)
    report_best_vrp(routes)

    # Local improvement: inter-route relocate and intra-route 2-opt
    max_iter = min(n * truck_count, 200)
    for _ in range(max_iter):
        improved = False
        # Inter-route relocate
        for i in range(truck_count):
            route_i = routes[i]
            if len(route_i) <= 2:
                continue
            for cust in route_i[1:-1]:
                for j in range(truck_count):
                    if i == j:
                        continue
                    route_j = routes[j]
                    # Try inserting cust into every position in route_j (excluding before 0 and after 0)
                    for pos in range(1, len(route_j)):
                        new_routes = [list(r) for r in routes]
                        new_routes[i].remove(cust)
                        new_routes[j].insert(pos, cust)
                        new_max = compute_max_distance(new_routes)
                        if new_max < best_max:
                            routes = new_routes
                            report_best_vrp(new_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Intra-route 2-opt for each route
        for i in range(truck_count):
            route = routes[i]
            if len(route) <= 3:
                continue
            best_gain = 0
            best_swapped = None
            for i1 in range(1, len(route)-2):
                for i2 in range(i1+1, len(route)-1):
                    new_route = route[:i1] + route[i1:i2+1][::-1] + route[i2+1:]
                    gain = route_distance(route) - route_distance(new_route)
                    if gain > best_gain:
                        best_gain = gain
                        best_swapped = new_route
            if best_gain > 0:
                routes[i] = best_swapped
                improved = True
                report_best_vrp(routes)
        if not improved:
            break
    # Final check: if best_routes is still None, return something
    if best_routes is None:
        return routes
    return best_routes