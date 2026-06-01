import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Farthest-first seed selection
    seeds = []
    seed0 = max(customers, key=lambda x: distance_matrix[0, x])
    seeds.append(seed0)
    while len(seeds) < truck_count:
        best_cust = None
        best_min_dist = -1.0
        for c in customers:
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c, s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and best_cust is not None and c < best_cust):
                best_min_dist = min_dist
                best_cust = c
        if best_cust is not None:
            seeds.append(best_cust)
        else:
            break

    # Nearest assignment to seeds
    clusters = [[] for _ in range(truck_count)]
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
    for i, s in enumerate(seeds):
        clusters[i].append(s)

    # Cheapest insertion to build routes for each cluster
    routes = []
    for cl in clusters:
        if not cl:
            routes.append([0, 0])
        else:
            unvisited = list(cl)
            route = [0, 0]
            while unvisited:
                best_cust = None
                best_pos = None
                best_cost = float('inf')
                for c in unvisited:
                    for pos in range(1, len(route)):
                        delta = (distance_matrix[route[pos-1], c] +
                                 distance_matrix[c, route[pos]] -
                                 distance_matrix[route[pos-1], route[pos]])
                        if delta < best_cost or (delta == best_cost and best_cust is not None and c < best_cust):
                            best_cost = delta
                            best_cust = c
                            best_pos = pos
                route.insert(best_pos, best_cust)
                unvisited.remove(best_cust)
            routes.append(route)
    report_best_vrp(routes)

    # Improvement loop
    max_iter = min(n * truck_count, 200)
    for _ in range(max_iter):
        # Find longest route
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        interior = routes[max_idx][1:-1]
        if not interior:
            break

        best_move = None
        best_new_max = float('inf')

        # Relocate moves from longest route to others
        for cust in sorted(interior):
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    new_routes = [list(r) for r in routes]
                    new_routes[max_idx].remove(cust)
                    new_routes[other_idx].insert(pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_move = ('relocate', cust, max_idx, other_idx, pos, new_routes)
                    elif abs(new_max - best_new_max) < 1e-12:
                        if cust < best_move[1] or (cust == best_move[1] and other_idx < best_move[3]) or (cust == best_move[1] and other_idx == best_move[3] and pos < best_move[4]):
                            best_new_max = new_max
                            best_move = ('relocate', cust, max_idx, other_idx, pos, new_routes)

        # Swap moves between longest route and others
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_interior = routes[other_idx][1:-1]
            if not other_interior:
                continue
            for cust_max in sorted(interior):
                for cust_other in sorted(other_interior):
                    new_routes = [list(r) for r in routes]
                    idx_max = new_routes[max_idx].index(cust_max)
                    idx_other = new_routes[other_idx].index(cust_other)
                    new_routes[max_idx][idx_max] = cust_other
                    new_routes[other_idx][idx_other] = cust_max
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_move = ('swap', cust_max, max_idx, cust_other, other_idx, new_routes)
                    elif abs(new_max - best_new_max) < 1e-12:
                        if cust_max < best_move[1] or (cust_max == best_move[1] and other_idx < best_move[4]) or (cust_max == best_move[1] and other_idx == best_move[4] and cust_other < best_move[3]):
                            best_new_max = new_max
                            best_move = ('swap', cust_max, max_idx, cust_other, other_idx, new_routes)

        if best_move is None:
            break
        # Apply best move
        if best_move[0] == 'relocate':
            _, cust, from_idx, to_idx, pos, new_routes = best_move
            routes = new_routes
        else:
            _, cust_max, max_idx, cust_other, other_idx, new_routes = best_move
            routes = new_routes
        report_best_vrp(routes)

        # Intra-route 2-opt on each route
        for i in range(truck_count):
            route = routes[i]
            if len(route) <= 3:
                continue
            improved = True
            while improved:
                improved = False
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        if route_distance(new_route) < route_distance(route) - 1e-12:
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
            routes[i] = route
            new_max = max(route_distance(r) for r in routes)
            if new_max < best_max - 1e-12:
                report_best_vrp(routes)

    report_best_vrp(routes)
    return best_routes if best_routes is not None else routes