import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = len(distance_matrix)
    customers = list(range(1, n))
    best_routes = None
    best_max_dist = float('inf')

    def route_distance(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total

    def compute_routes_and_max(clusters):
        routes = []
        for cl in clusters:
            if not cl:
                routes.append([0,0])
            else:
                route = [0]
                unvisited = list(cl)
                current = 0
                while unvisited:
                    nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
                    route.append(nearest)
                    unvisited.remove(nearest)
                    current = nearest
                route.append(0)
                routes.append(route)
        distances = [route_distance(r) for r in routes]
        return routes, max(distances)

    def report_best_vrp(routes):
        nonlocal best_routes, best_max_dist
        max_d = max(route_distance(r) for r in routes)
        if max_d < best_max_dist:
            best_max_dist = max_d
            best_routes = [list(r) for r in routes]

    # Special case: truck_count >= number of customers
    if truck_count >= n - 1:
        clusters = [[c] for c in customers] + [[] for _ in range(truck_count - (n - 1))]
        routes, _ = compute_routes_and_max(clusters)
        report_best_vrp(routes)
        return best_routes

    # Step 1: Farthest-first seed selection
    seeds = []
    seed0 = max(customers, key=lambda x: distance_matrix[0][x])
    seeds.append(seed0)
    while len(seeds) < truck_count:
        best_customer = None
        best_min_dist = -1
        for c in customers:
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c][s] for s in seeds)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_customer = c
            elif min_dist == best_min_dist and best_customer is not None and c < best_customer:
                best_customer = c
        if best_customer is not None:
            seeds.append(best_customer)
        else:
            break

    # Step 2: Assign customers to nearest seed
    clusters = [[] for _ in range(truck_count)]
    for c in customers:
        if c in seeds:
            continue
        min_dist = float('inf')
        best_idx = 0
        for i, s in enumerate(seeds):
            d = distance_matrix[c][s]
            if d < min_dist or (d == min_dist and i < best_idx):
                min_dist = d
                best_idx = i
        clusters[best_idx].append(c)
    for i, s in enumerate(seeds):
        clusters[i].append(s)

    # Build initial routes
    routes, _ = compute_routes_and_max(clusters)
    report_best_vrp(routes)

    # Step 3: Greedy balancing - move customer that gives maximum reduction in max distance
    max_iter = n * truck_count
    for _ in range(max_iter):
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        custs = [c for c in routes[max_idx] if c != 0]
        if not custs:
            break
        best_move = None
        best_new_max = best_max_dist
        for c in custs:
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                new_clusters = [list(clusters[i]) for i in range(truck_count)]
                new_clusters[max_idx].remove(c)
                new_clusters[other_idx].append(c)
                new_routes, new_max = compute_routes_and_max(new_clusters)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = (c, other_idx, new_clusters, new_routes)
        if best_move is not None:
            clusters = best_move[2]
            routes = best_move[3]
            report_best_vrp(routes)
        else:
            break

    # Step 4: Local 2-opt improvement per route (single pass best improvement, bounded)
    for i in range(truck_count):
        route = routes[i]
        if len(route) <= 3:
            continue
        improved = True
        max_passes = 10  # safety bound
        for _ in range(max_passes):
            best_gain = 0
            best_swap = None
            for i1 in range(1, len(route)-2):
                for i2 in range(i1+1, len(route)-1):
                    new_route = route[:i1] + route[i1:i2+1][::-1] + route[i2+1:]
                    gain = route_distance(route) - route_distance(new_route)
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = new_route
            if best_gain > 0:
                route = best_swap
                routes[i] = route
            else:
                break
    # Final check
    max_d = max(route_distance(r) for r in routes)
    if max_d < best_max_dist:
        best_routes = [list(r) for r in routes]
    return best_routes