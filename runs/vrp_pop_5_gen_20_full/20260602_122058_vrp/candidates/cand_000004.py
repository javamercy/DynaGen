def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # Seed selection: farthest-first from depot
    seeds = []
    first_seed = max(range(1, n), key=lambda i: (distance_matrix[0][i], -i))
    seeds.append(first_seed)
    for _ in range(1, truck_count):
        best_min_dist = -1
        best_node = None
        for node in range(1, n):
            if node in seeds:
                continue
            min_dist = min(distance_matrix[node][s] for s in seeds)
            if min_dist > best_min_dist or (min_dist == best_min_dist and node < best_node):
                best_min_dist = min_dist
                best_node = node
        if best_node is None:
            break
        seeds.append(best_node)

    # Assign customers to nearest seed
    clusters = {s: [] for s in seeds}
    for node in range(1, n):
        if node in seeds:
            clusters[node].append(node)
        else:
            nearest = min(seeds, key=lambda s: (distance_matrix[node][s], s))
            clusters[nearest].append(node)

    # Build initial routes using nearest neighbor from depot
    routes = []
    for seed in seeds:
        cluster = clusters[seed]
        unvisited = set(cluster)
        route = [0]
        current = 0
        while unvisited:
            next_node = min(unvisited, key=lambda x: (distance_matrix[current][x], x))
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        route.append(0)
        routes.append(route)

    while len(routes) < truck_count:
        routes.append([0, 0])

    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d

    max_dist = max(route_dist(r) for r in routes)
    report_best_vrp(routes)

    # Inter-route improvement
    max_iter = n * truck_count
    for _ in range(max_iter):
        improved = False
        # Relocate a customer to another route
        for i in range(1, n):
            # Find current route and position of i
            curr_route_idx = None
            pos = None
            for idx, route in enumerate(routes):
                if i in route:
                    curr_route_idx = idx
                    pos = route.index(i)
                    break
            if curr_route_idx is None:
                continue
            curr_route = routes[curr_route_idx]
            for t_idx in range(truck_count):
                if t_idx == curr_route_idx:
                    continue
                target_route = routes[t_idx]
                for ins_pos in range(1, len(target_route)):
                    new_curr = curr_route[:pos] + curr_route[pos+1:]
                    new_target = target_route[:ins_pos] + [i] + target_route[ins_pos:]
                    old_max = max(route_dist(curr_route), route_dist(target_route))
                    new_max = max(route_dist(new_curr), route_dist(new_target))
                    # Recompute overall max
                    current_max = max(route_dist(r) for r in routes)
                    if new_max < current_max:
                        routes[curr_route_idx] = new_curr
                        routes[t_idx] = new_target
                        report_best_vrp(routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Exchange customers between two routes
        for i in range(1, n):
            ri = None
            for idx, route in enumerate(routes):
                if i in route:
                    ri = idx
                    break
            if ri is None:
                continue
            route_i = routes[ri]
            for j in range(i+1, n):
                rj = None
                for idx, route in enumerate(routes):
                    if j in route:
                        rj = idx
                        break
                if rj is None or rj == ri:
                    continue
                route_j = routes[rj]
                pi = route_i.index(i)
                pj = route_j.index(j)
                new_i = route_i[:]
                new_i[pi] = j
                new_j = route_j[:]
                new_j[pj] = i
                old_max = max(route_dist(route_i), route_dist(route_j))
                new_max = max(route_dist(new_i), route_dist(new_j))
                current_max = max(route_dist(r) for r in routes)
                if new_max < current_max:
                    routes[ri] = new_i
                    routes[rj] = new_j
                    report_best_vrp(routes)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return routes