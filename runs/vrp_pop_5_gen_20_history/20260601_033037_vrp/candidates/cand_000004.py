import numpy as np
import heapq
import itertools

def solve_vrp(distance_matrix, truck_count):
    n = len(distance_matrix)
    customers = list(range(1, n))
    best_routes = None
    best_max_dist = float('inf')

    def route_distance(route):
        if len(route) <= 2:
            # route is [0,0] or [0, cust, 0]
            return 2 * distance_matrix[0][0]  # actually zero if [0,0], else compute
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
                # nearest neighbor from depot
                route = [0]
                unvisited = list(cl)
                current = 0
                while unvisited:
                    # find nearest unvisited
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

    # Initial clustering
    if truck_count >= n:
        # each customer its own route, rest empty
        clusters = [[c] for c in customers] + [[] for _ in range(truck_count - len(customers))]
        routes, _ = compute_routes_and_max(clusters)
        report_best_vrp(routes)
        return best_routes

    # farthest-first seed selection
    seeds = []
    # first seed: farthest from depot
    seed0 = max(customers, key=lambda x: distance_matrix[0][x])
    seeds.append(seed0)
    while len(seeds) < truck_count:
        # for each customer, min distance to any seed
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
                # tie break by smaller index
                best_customer = c
        if best_customer is not None:
            seeds.append(best_customer)
        else:
            break
    # assign customers to nearest seed
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
    # add seeds to their clusters
    for i, s in enumerate(seeds):
        clusters[i].append(s)
    # initial routes
    routes, _ = compute_routes_and_max(clusters)
    report_best_vrp(routes)

    # Improvement: attempt to move a customer from the route with max distance to another
    # Loop limited to number of customers * truck_count iterations
    for _ in range(min(n * truck_count, 100)):
        # find route with max distance and candidates
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        # customers in that route (excluding depot)
        custs = [c for c in routes[max_idx] if c != 0]
        if not custs:
            break
        # try moving each customer to each other route
        improved = False
        for c in custs:
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                # hypothetical new clusters
                new_clusters = [list(clusters[i]) for i in range(truck_count)]
                new_clusters[max_idx].remove(c)
                new_clusters[other_idx].append(c)
                new_routes, new_max = compute_routes_and_max(new_clusters)
                if new_max < best_max_dist:
                    clusters = new_clusters
                    routes = new_routes
                    report_best_vrp(routes)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Improve each route individually with 2-opt (limited)
    for i in range(truck_count):
        route = routes[i]
        if len(route) <= 3:
            continue
        improved = True
        iter_count = 0
        while improved and iter_count < 10:
            improved = False
            for a, b, c, d in itertools.combinations(range(1, len(route)-1), 4):
                # edges (a-1,a) and (c-1,c) reversed? Actually 2-opt: replace (a-1,a) and (b,b+1) with (a-1,b) and (a,b+1)
                # We'll iterate over all pairs of edges
                pass
            # Simplified: just do one pass of neighbor swapping? Better to do a finite 2-opt
            # Use indices for simplicity
            best_swap = None
            best_gain = 0
            for i1 in range(1, len(route)-2):
                for i2 in range(i1+1, len(route)-1):
                    # reverse segment from i1 to i2
                    new_route = route[:i1] + route[i1:i2+1][::-1] + route[i2+1:]
                    gain = route_distance(route) - route_distance(new_route)
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = new_route
            if best_gain > 0:
                routes[i] = best_swap
                improved = True
            iter_count += 1
        # update cluster list? Not needed for reporting final
    # Update cluster list to match routes (for consistency, but not necessary)
    # Recompute max distance for final
    max_d = max(route_distance(r) for r in routes)
    if max_d < best_max_dist:
        best_routes = [list(r) for r in routes]
    return best_routes