import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    num_customers = n - 1
    if truck_count < 1:
        return []
    customers = list(range(1, n))
    if truck_count >= num_customers:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes[:truck_count]

    # Farthest-first seeding
    seeds = []
    # first seed: farthest from depot
    first_seed = max(range(1, n), key=lambda c: distance_matrix[0][c])
    seeds.append(first_seed)
    while len(seeds) < truck_count:
        max_min_dist = -1
        best_cust = None
        for c in range(1, n):
            if c in seeds:
                continue
            min_dist = min(distance_matrix[c][s] for s in seeds)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_cust = c
            elif min_dist == max_min_dist and best_cust is not None and c < best_cust:
                best_cust = c
        seeds.append(best_cust)
    # sort seeds for determinism
    seeds.sort()

    # Assign customers to nearest seed
    clusters = {i: [] for i in range(truck_count)}
    for c in range(1, n):
        min_dist = float('inf')
        best_idx = None
        for idx, s in enumerate(seeds):
            d = distance_matrix[c][s]
            if d < min_dist:
                min_dist = d
                best_idx = idx
            elif d == min_dist and best_idx is not None and idx < best_idx:
                best_idx = idx
        clusters[best_idx].append(c)

    # Build routes with cheapest insertion
    def route_dist(r):
        d = 0
        for a, b in zip(r, r[1:]):
            d += distance_matrix[a][b]
        return d

    routes = []
    for idx in range(truck_count):
        cluster = clusters[idx]
        if not cluster:
            routes.append([0, 0])
            continue
        # Start with depot
        route = [0]
        # Insert customers one by one using cheapest insertion
        remaining = cluster[:]
        while remaining:
            best_increase = float('inf')
            best_pos = None
            best_cust = None
            for cust in remaining:
                for pos in range(1, len(route)+1):
                    # new route: route[:pos] + [cust] + route[pos:]
                    new_route = route[:pos] + [cust] + route[pos:]
                    # compute increase relative to original route + return to depot
                    # actually we need to compute total dist of new_route + final return?
                    # We'll compute total distance of new_route plus return to depot if not already ending at depot
                    new_dist = 0
                    for a, b in zip(new_route, new_route[1:]):
                        new_dist += distance_matrix[a][b]
                    new_dist += distance_matrix[new_route[-1]][0]  # return to depot
                    # Original route distance without depot return? Since depot is included at start, original route ends at last customer, need to add return to depot
                    original_dist = 0
                    for a, b in zip(route, route[1:]):
                        original_dist += distance_matrix[a][b]
                    original_dist += distance_matrix[route[-1]][0]
                    increase = new_dist - original_dist
                    if increase < best_increase:
                        best_increase = increase
                        best_pos = pos
                        best_cust = cust
                    elif increase == best_increase and best_cust is not None and cust < best_cust:
                        best_pos = pos
                        best_cust = cust
            # Insert best_cust at best_pos in route, and close with depot? We'll keep route without depot at end until final
            route = route[:best_pos] + [best_cust] + route[best_pos:]
            remaining.remove(best_cust)
        # Close route with depot
        route.append(0)
        routes.append(route)

    # Ensure exactly truck_count routes
    while len(routes) < truck_count:
        routes.append([0, 0])
    routes = routes[:truck_count]

    # Intra-route 2-opt on each route
    for idx in range(len(routes)):
        improved = True
        max_iter_inner = num_customers
        iter_count = 0
        while improved and iter_count < max_iter_inner:
            improved = False
            iter_count += 1
            route = routes[idx]
            best_delta = 0
            best_i = best_j = None
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # compute delta
                    old1 = distance_matrix[route[i-1]][route[i]]
                    old2 = distance_matrix[route[j]][route[j+1]]
                    new1 = distance_matrix[route[i-1]][route[j]]
                    new2 = distance_matrix[route[i]][route[j+1]]
                    delta = (new1 + new2) - (old1 + old2)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_delta < 0:
                route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                routes[idx] = route
                improved = True
                report_best_vrp(routes)

    def route_dist(r):
        d = 0
        for a, b in zip(r, r[1:]):
            d += distance_matrix[a][b]
        return d

    report_best_vrp(routes)

    # Adaptive local search with 2-opt on longest route
    max_iter = max(20, num_customers * 2)
    for iteration in range(max_iter):
        longest_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        current_max = route_dist(routes[longest_idx])
        best_new_routes = None
        best_new_max = current_max

        # Try 2-opt on longest route
        route = routes[longest_idx]
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_routes = routes[:]
                new_routes[longest_idx] = new_route
                new_max = max(route_dist(r) for r in new_routes)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_new_routes = new_routes

        # Try moves and swaps from longest route
        for pos, cust in enumerate(routes[longest_idx][1:-1]):
            # Move to another route
            new_longest_route = routes[longest_idx][:pos+1] + routes[longest_idx][pos+2:]
            for other_idx in range(len(routes)):
                if other_idx == longest_idx:
                    continue
                other_route = routes[other_idx]
                for ins_pos in range(1, len(other_route)):
                    new_other_route = other_route[:ins_pos] + [cust] + other_route[ins_pos:]
                    new_routes = routes[:]
                    new_routes[longest_idx] = new_longest_route
                    new_routes[other_idx] = new_other_route
                    new_max = max(route_dist(r) for r in new_routes)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_new_routes = new_routes
            # Swap with customer from another route
            for other_idx in range(len(routes)):
                if other_idx == longest_idx:
                    continue
                other_route = routes[other_idx]
                for opos, ocust in enumerate(other_route[1:-1]):
                    # Swap cust and ocust
                    # Build new routes
                    new_longest_route = routes[longest_idx][:]
                    new_longest_route[pos+1] = ocust
                    new_other_route = other_route[:]
                    new_other_route[opos+1] = cust
                    new_routes = routes[:]
                    new_routes[longest_idx] = new_longest_route
                    new_routes[other_idx] = new_other_route
                    new_max = max(route_dist(r) for r in new_routes)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_new_routes = new_routes

        if best_new_routes is not None and best_new_max < current_max:
            routes = best_new_routes
            report_best_vrp(routes)
        else:
            # No improvement found
            pass

    # Post-search balancing: move customers from longest to shortest
    balance_iters = num_customers // truck_count
    for _ in range(balance_iters):
        longest_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        shortest_idx = min(range(len(routes)), key=lambda i: route_dist(routes[i]))
        if shortest_idx == longest_idx:
            break
        best_max = route_dist(routes[longest_idx])
        best_move = None
        longest_route = routes[longest_idx]
        shortest_route = routes[shortest_idx]
        for pos, cust in enumerate(longest_route[1:-1]):
            new_longest = longest_route[:pos+1] + longest_route[pos+2:]
            for ins_pos in range(1, len(shortest_route)):
                new_shortest = shortest_route[:ins_pos] + [cust] + shortest_route[ins_pos:]
                new_max = max(route_dist(new_longest), route_dist(new_shortest))
                if new_max < best_max:
                    best_max = new_max
                    best_move = (pos, ins_pos, cust)
        if best_move is not None:
            pos, ins_pos, cust = best_move
            routes[longest_idx] = longest_route[:pos+1] + longest_route[pos+2:]
            routes[shortest_idx] = shortest_route[:ins_pos] + [cust] + shortest_route[ins_pos:]
            report_best_vrp(routes)

    # Final 2-opt on all routes (optional but helpful)
    for idx in range(len(routes)):
        improved = True
        max_iter_inner = num_customers
        iter_count = 0
        while improved and iter_count < max_iter_inner:
            improved = False
            iter_count += 1
            route = routes[idx]
            best_delta = 0
            best_i = best_j = None
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old1 = distance_matrix[route[i-1]][route[i]]
                    old2 = distance_matrix[route[j]][route[j+1]]
                    new1 = distance_matrix[route[i-1]][route[j]]
                    new2 = distance_matrix[route[i]][route[j+1]]
                    delta = (new1 + new2) - (old1 + old2)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_delta < 0:
                route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                routes[idx] = route
                improved = True
                report_best_vrp(routes)

    return routes