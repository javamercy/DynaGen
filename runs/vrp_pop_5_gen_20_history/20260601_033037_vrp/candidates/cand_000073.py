import numpy as np
import heapq

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

    # Nearest neighbor insertion construction
    unvisited = set(customers)
    route_list = [[0, 0] for _ in range(truck_count)]
    # Round-robin insertion
    round_idx = 0
    while unvisited:
        idx = round_idx % truck_count
        route = route_list[idx]
        # Find best customer and position for insertion
        best_delta = float('inf')
        best_cust = None
        best_pos = None
        for cust in sorted(unvisited):  # deterministic order
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_cust = cust
                    best_pos = pos
        if best_cust is not None:
            route.insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        round_idx += 1

    report_best_vrp(route_list)

    max_iter = min(300, n * truck_count)
    for _ in range(max_iter):
        improved = False
        dists = [route_distance(r) for r in route_list]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        interior = route_list[max_idx][1:-1]
        # Inter-route relocate from longest route
        if interior:
            for cust in interior:
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = route_list[other_idx]
                    best_pos = None
                    best_delta = float('inf')
                    for pos in range(1, len(other_route)):
                        prev = other_route[pos-1]
                        nxt = other_route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = pos
                    new_routes = [list(r) for r in route_list]
                    new_routes[max_idx].remove(cust)
                    new_routes[other_idx].insert(best_pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_max - 1e-12:
                        route_list = new_routes
                        report_best_vrp(route_list)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
        # Inter-route swap between longest and another route
        if interior:
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_interior = route_list[other_idx][1:-1]
                if not other_interior:
                    continue
                for cust_max in interior:
                    for cust_other in other_interior:
                        new_routes = [list(r) for r in route_list]
                        idx_max = new_routes[max_idx].index(cust_max)
                        idx_other = new_routes[other_idx].index(cust_other)
                        new_routes[max_idx][idx_max] = cust_other
                        new_routes[other_idx][idx_other] = cust_max
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max - 1e-12:
                            route_list = new_routes
                            report_best_vrp(route_list)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
        # Intra-route 2-opt
        for idx in range(truck_count):
            route = route_list[idx]
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_dist = route_distance(route)
            found = False
            for a in range(1, len(route)-2):
                for b in range(a+1, len(route)-1):
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - 1e-12:
                        best_dist = new_dist
                        best_route = new_route
                        found = True
                        break
                if found:
                    break
            if found:
                route_list[idx] = best_route
                new_max = max(route_distance(r) for r in route_list)
                if new_max < best_max - 1e-12:
                    report_best_vrp(route_list)
                improved = True
                break
        if not improved:
            break

    final_routes = best_routes if best_routes is not None else route_list
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes