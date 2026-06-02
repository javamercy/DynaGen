import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # --- Initial Construction: farthest-first clustering + nearest neighbor ordering ---
    # Seed selection: farthest from depot
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

    # Build routes using nearest neighbor from depot
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

    def route_dist(r):
        d = 0
        for i in range(len(r)-1):
            d += distance_matrix[r[i]][r[i+1]]
        return d

    current_max = max(route_dist(r) for r in routes)
    report_best_vrp(routes)

    # --- Threshold Accepting Improvement ---
    max_iter = n * truck_count  # finite bound
    threshold = 0.3 * current_max
    cooling = 0.95
    for iteration in range(max_iter):
        improved = False
        # Find longest route
        max_dist = current_max
        max_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        long_route = routes[max_idx]
        # Relocate customers from longest route
        for pos, cust in enumerate(long_route[1:-1]):
            new_long = long_route[:pos+1] + long_route[pos+2:]
            for t_idx in range(truck_count):
                if t_idx == max_idx:
                    continue
                target_route = routes[t_idx]
                for ins_pos in range(1, len(target_route)):
                    new_target = target_route[:ins_pos] + [cust] + target_route[ins_pos:]
                    new_max_val = max(route_dist(new_long), route_dist(new_target))
                    for r_idx in range(truck_count):
                        if r_idx not in (max_idx, t_idx):
                            new_max_val = max(new_max_val, route_dist(routes[r_idx]))
                    if new_max_val <= current_max + threshold:
                        if new_max_val < current_max:
                            current_max = new_max_val
                            routes[max_idx] = new_long
                            routes[t_idx] = new_target
                            report_best_vrp(routes)
                            improved = True
                            break
                        elif new_max_val <= current_max + threshold:
                            routes[max_idx] = new_long
                            routes[t_idx] = new_target
                            current_max = new_max_val
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            threshold *= cooling
            continue
        # Exchange customers between longest route and others
        for pos, cust in enumerate(long_route[1:-1]):
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for opos, ocust in enumerate(other_route[1:-1]):
                    new_long = long_route[:pos+1] + [ocust] + long_route[pos+2:]
                    new_other = other_route[:opos+1] + [cust] + other_route[opos+2:]
                    new_max_val = max(route_dist(new_long), route_dist(new_other))
                    for r_idx in range(truck_count):
                        if r_idx not in (max_idx, other_idx):
                            new_max_val = max(new_max_val, route_dist(routes[r_idx]))
                    if new_max_val <= current_max + threshold:
                        if new_max_val < current_max:
                            current_max = new_max_val
                            routes[max_idx] = new_long
                            routes[other_idx] = new_other
                            report_best_vrp(routes)
                            improved = True
                            break
                        else:
                            routes[max_idx] = new_long
                            routes[other_idx] = new_other
                            current_max = new_max_val
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
        threshold *= cooling

    # Intra-route 2-opt for each route
    for r_idx in range(truck_count):
        route = routes[r_idx]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    a = route[i-1]
                    b = route[i]
                    c = route[j]
                    d = route[j+1]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new_dist = distance_matrix[a][c] + distance_matrix[b][d]
                    if new_dist < old - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break
    return routes