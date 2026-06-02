import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # --- Initial Construction: best insertion with min-max objective ---
    routes = [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    # sort customers by decreasing distance from depot (ties: lower index)
    customers.sort(key=lambda i: (-distance_matrix[0][i], i))

    def route_dist(r):
        d = 0
        for i in range(len(r)-1):
            d += distance_matrix[r[i]][r[i+1]]
        return d

    for cust in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_dist(new_route)
                # compute max over all routes with this insertion
                new_max = new_dist
                for other_idx in range(truck_count):
                    if other_idx != r_idx:
                        new_max = max(new_max, route_dist(routes[other_idx]))
                if new_max < best_max or (new_max == best_max and (r_idx < best_route_idx or (r_idx == best_route_idx and pos < best_pos))):
                    best_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]

    current_max = max(route_dist(r) for r in routes)
    report_best_vrp(routes)

    # --- Threshold Accepting Improvement ---
    max_iter = n * truck_count  # finite bound
    threshold = 0.3 * current_max
    cooling = 0.95
    for iteration in range(max_iter):
        improved = False
        # Find longest route
        max_idx = max(range(truck_count), key=lambda i: route_dist(routes[i]))
        long_route = routes[max_idx]
        max_dist = current_max
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