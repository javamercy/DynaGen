import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # Clarke-Wright savings initialization
    routes = [[0, i, 0] for i in range(1, n)]
    if len(routes) > truck_count:
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append((s, i, j))
        savings.sort(key=lambda x: (-x[0], x[1], x[2]))  # deterministic tie-break

        for s, i, j in savings:
            if len(routes) == truck_count:
                break
            idx_i = idx_j = -1
            for idx, route in enumerate(routes):
                if i in route:
                    idx_i = idx
                if j in route:
                    idx_j = idx
            if idx_i == idx_j or idx_i == -1 or idx_j == -1:
                continue
            route_i = routes[idx_i]
            route_j = routes[idx_j]
            # Check if i is at end of route_i and j at start of route_j
            if route_i[-2] == i and route_j[1] == j:
                new_route = route_i[:-1] + route_j[1:]
                if idx_i < idx_j:
                    routes[idx_i] = new_route
                    routes.pop(idx_j)
                else:
                    routes[idx_j] = new_route
                    routes.pop(idx_i)
            elif route_j[-2] == j and route_i[1] == i:
                new_route = route_j[:-1] + route_i[1:]
                if idx_i < idx_j:
                    routes[idx_j] = new_route
                    routes.pop(idx_i)
                else:
                    routes[idx_i] = new_route
                    routes.pop(idx_j)

    while len(routes) < truck_count:
        routes.append([0, 0])

    # Compute initial best
    current_dist = [route_distance(r) for r in routes]
    best_routes = [list(r) for r in routes]
    best_max = max(current_dist)
    report_best_vrp(best_routes)

    # Local search
    customers = list(range(1, n))
    max_iters = 10 * len(customers) * truck_count
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1

        # Relocate
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for idx in range(1, len(route1)-1):
                cust = route1[idx]
                new_route1 = route1[:idx] + route1[idx+1:]
                new_dist1 = route_distance(new_route1)
                for r2 in range(truck_count):
                    if r2 == r1:
                        continue
                    route2 = routes[r2]
                    old_dist2 = current_dist[r2]
                    # Find best insertion position
                    best_cost = float('inf')
                    best_pos = -1
                    for p in range(1, len(route2)):
                        cost = distance_matrix[route2[p-1], cust] + distance_matrix[cust, route2[p]] - distance_matrix[route2[p-1], route2[p]]
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = p
                    new_route2 = route2[:best_pos] + [cust] + route2[best_pos:]
                    new_dist2 = old_dist2 + best_cost
                    other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1, r2)]
                    new_max = max(new_dist1, new_dist2, *other_dists)
                    if new_max < best_max:
                        routes[r1] = new_route1
                        routes[r2] = new_route2
                        current_dist[r1] = new_dist1
                        current_dist[r2] = new_dist2
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Swap
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust1 = route1[idx1]
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 2:
                        continue
                    for idx2 in range(1, len(route2)-1):
                        cust2 = route2[idx2]
                        new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                        new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1, r2)]
                        new_max = max(new_dist1, new_dist2, *other_dists)
                        if new_max < best_max:
                            routes[r1] = new_route1
                            routes[r2] = new_route2
                            current_dist[r1] = new_dist1
                            current_dist[r2] = new_dist2
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
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

        # Intra-route 2-opt
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    other_dists = [current_dist[k] for k in range(truck_count) if k != r]
                    new_max = max(new_dist, *other_dists)
                    if new_max < best_max:
                        routes[r] = new_route
                        current_dist[r] = new_dist
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Cross-route 2-opt
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for r2 in range(r1+1, truck_count):
                route2 = routes[r2]
                if len(route2) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        new1 = route1[:i+1] + route2[j+1:]
                        new2 = route2[:j+1] + route1[i+1:]
                        new_dist1 = route_distance(new1)
                        new_dist2 = route_distance(new2)
                        other_dists = [current_dist[k] for k in range(truck_count) if k not in (r1, r2)]
                        new_max = max(new_dist1, new_dist2, *other_dists)
                        if new_max < best_max:
                            routes[r1] = new1
                            routes[r2] = new2
                            current_dist[r1] = new_dist1
                            current_dist[r2] = new_dist2
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_routes