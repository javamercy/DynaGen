import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()
    customers = list(range(1, n))
    customers.sort(key=lambda c: dist[0][c])
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count
    current_max = 0.0
    total_cust = len(customers)

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += dist[route[i]][route[i+1]]
        return total

    # Insertion phase with adaptive weight
    for idx, cust in enumerate(customers):
        # Linear decrease from 1.0 to 0.2
        progress = idx / (total_cust - 1) if total_cust > 1 else 1.0
        lambda_weight = 1.0 - 0.8 * progress  # 1.0 to 0.2
        best_new_max = float('inf')
        best_delta = float('inf')
        best_route_len = float('inf')
        best_r = -1
        best_p = -1
        for r in range(truck_count):
            route = routes[r]
            for p in range(1, len(route)):
                delta = dist[route[p-1]][cust] + dist[cust][route[p]] - dist[route[p-1]][route[p]]
                new_len = route_lengths[r] + delta
                new_max = max(current_max, new_len)
                weighted_delta = delta * (1 + lambda_weight * (route_lengths[r] / (current_max + 1e-12)))
                if (new_max < best_new_max or
                    (new_max == best_new_max and weighted_delta < best_delta) or
                    (new_max == best_new_max and weighted_delta == best_delta and route_lengths[r] < best_route_len) or
                    (new_max == best_new_max and weighted_delta == best_delta and route_lengths[r] == best_route_len and r < best_r) or
                    (new_max == best_new_max and weighted_delta == best_delta and route_lengths[r] == best_route_len and r == best_r and p < best_p)):
                    best_new_max = new_max
                    best_delta = weighted_delta
                    best_route_len = route_lengths[r]
                    best_r = r
                    best_p = p
        routes[best_r].insert(best_p, cust)
        # Revert to actual delta (weighted_delta divided by factor)
        factor = 1 + lambda_weight * (route_lengths[best_r] / (current_max + 1e-12))
        route_lengths[best_r] += (delta_actual := best_delta / factor)
        current_max = max(current_max, route_lengths[best_r])

    # Update route lengths
    for r in range(truck_count):
        route_lengths[r] = route_distance(routes[r])
    current_max = max(route_lengths)
    report_best_vrp(routes)

    # Intra-route 2-opt with adaptive iterations
    max_iter = max(5, n // 5)
    for _ in range(max_iter):
        improved = False
        for r in range(truck_count):
            route = routes[r]
            best_route = route[:]
            best_len = route_lengths[r]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_len = route_distance(new_route)
                    if new_len < best_len - 1e-12:
                        best_route = new_route
                        best_len = new_len
                        improved = True
            if best_len < route_lengths[r] - 1e-12:
                routes[r] = best_route
                route_lengths[r] = best_len
                current_max = max(route_lengths)
                report_best_vrp(routes)
        if not improved:
            break

    # Inter-route relocate with adaptive iterations
    for _ in range(max_iter):
        improved = False
        longest_idx = max(range(truck_count), key=lambda r: route_lengths[r])
        longest_len = route_lengths[longest_idx]
        route = routes[longest_idx]
        for idx, cust in enumerate(route):
            if cust == 0:
                continue
            new_route_long = route[:idx] + route[idx+1:]
            if len(new_route_long) < 2:
                continue
            new_len_long = route_distance(new_route_long)
            for r2 in range(truck_count):
                if r2 == longest_idx:
                    continue
                route2 = routes[r2]
                for p in range(1, len(route2)):
                    delta = dist[route2[p-1]][cust] + dist[cust][route2[p]] - dist[route2[p-1]][route2[p]]
                    new_len2 = route_lengths[r2] + delta
                    other_lengths = [route_lengths[rr] for rr in range(truck_count) if rr != longest_idx and rr != r2]
                    new_max = max(new_len_long, new_len2, *other_lengths)
                    if new_max < current_max - 1e-12:
                        new_route2 = route2[:p] + [cust] + route2[p:]
                        routes[longest_idx] = new_route_long
                        routes[r2] = new_route2
                        route_lengths[longest_idx] = new_len_long
                        route_lengths[r2] = new_len2
                        current_max = new_max
                        report_best_vrp(routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    report_best_vrp(routes)
    return routes