import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = len(distance_matrix)
    customers = list(range(1, n))
    best_routes = None
    best_max_dist = float('inf')

    def route_distance(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

    def compute_routes(clusters):
        routes = []
        for cl in clusters:
            if not cl:
                routes.append([0, 0])
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
        return routes

    def report_best(routes):
        nonlocal best_routes, best_max_dist
        max_d = max(route_distance(r) for r in routes)
        if max_d < best_max_dist:
            best_max_dist = max_d
            best_routes = [list(r) for r in routes]

    # Special case: enough trucks for each customer
    if truck_count >= n - 1:
        clusters = [[c] for c in customers] + [[] for _ in range(truck_count - (n - 1))]
        routes = compute_routes(clusters)
        report_best(routes)
        return best_routes

    # Farthest-first seed selection
    seeds = []
    first_seed = max(customers, key=lambda x: distance_matrix[0][x])
    seeds.append(first_seed)
    while len(seeds) < truck_count:
        best_cust = None
        best_min_dist = -1.0
        for c in customers:
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c][s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and (best_cust is None or c < best_cust)):
                best_min_dist = min_dist
                best_cust = c
        if best_cust is not None:
            seeds.append(best_cust)
        else:
            break

    # Assign each non-seed customer to nearest seed (tie-break smaller seed index)
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

    routes = compute_routes(clusters)
    report_best(routes)

    # Balancing improvement: relocate from longest route
    max_iter = n * truck_count
    for _ in range(max_iter):
        current_dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(current_dists)), key=lambda i: (current_dists[i], -i))
        longest_route = routes[max_idx]
        custs = [c for c in longest_route if c != 0]
        best_move = None
        best_new_max = best_max_dist
        for c in custs:
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                # Find best insertion position in other_route (min insertion cost)
                best_pos = None
                best_delta = float('inf')
                for pos in range(1, len(other_route)):
                    prev = other_route[pos-1]
                    nxt = other_route[pos] if pos < len(other_route)-1 else 0
                    d_ins = distance_matrix[prev][c] + distance_matrix[c][nxt] - distance_matrix[prev][nxt]
                    if d_ins < best_delta:
                        best_delta = d_ins
                        best_pos = pos
                delta = best_delta
                # Compute new routes
                new_routes = [list(r) for r in routes]
                new_routes[max_idx].remove(c)
                new_routes[other_idx].insert(best_pos, c)
                new_max = max(route_distance(r) for r in new_routes)
                if new_max < best_new_max or (new_max == best_new_max and (c < (best_move[0] if best_move else float('inf')) or (c == best_move[0] and other_idx < best_move[1]))):
                    best_new_max = new_max
                    best_move = (c, other_idx, new_routes)
        if best_move is not None and best_new_max < best_max_dist:
            routes = best_move[2]
            report_best(routes)
        else:
            break

    return best_routes