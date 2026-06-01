import numpy as np
from collections import deque

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

    # Construction: min-max insertion
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(customers)
    while unassigned:
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_new_max = float('inf')
        sorted_cust = sorted(unassigned)
        for cust in sorted_cust:
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_route_dist = route_distance(new_route)
                    temp_routes = [list(r) for r in routes]
                    temp_routes[r_idx] = new_route
                    new_max = max(route_distance(r) for r in temp_routes)
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_cust = cust
                        best_route_idx = r_idx
                        best_pos = pos
                    elif abs(new_max - best_new_max) < 1e-12:
                        if cust < best_cust or (cust == best_cust and r_idx < best_route_idx) or \
                           (cust == best_cust and r_idx == best_route_idx and pos < best_pos):
                            best_new_max = new_max
                            best_cust = cust
                            best_route_idx = r_idx
                            best_pos = pos
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    report_best_vrp(routes)

    # Iterative local search: repeatedly improve via relocate and 2-opt
    improved = True
    while improved:
        improved = False
        # Find longest route
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        route_max = routes[max_idx]
        interior = route_max[1:-1]
        if not interior:
            break

        # Relocate moves from max route to other routes
        best_move = None
        best_new_max = float('inf')
        for cust in interior:
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
                        if cust < best_move[1] if best_move else True:
                            best_new_max = new_max
                            best_move = ('relocate', cust, max_idx, other_idx, pos, new_routes)

        # Swap moves between max route and other routes
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_interior = routes[other_idx][1:-1]
            if not other_interior:
                continue
            for cust_max in interior:
                for cust_other in other_interior:
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
                        if cust_max < best_move[1] if best_move else True:
                            best_new_max = new_max
                            best_move = ('swap', cust_max, max_idx, cust_other, other_idx, new_routes)

        if best_move is not None and best_new_max < best_max - 1e-12:
            # Improvement found, apply move
            if best_move[0] == 'relocate':
                _, cust, from_idx, to_idx, pos, new_routes = best_move
            else:
                _, cust_max, max_idx, cust_other, other_idx, new_routes = best_move
            routes = new_routes
            report_best_vrp(routes)
            improved = True

        # Intra-route 2-opt on each route after move
        for idx in range(truck_count):
            route = routes[idx]
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
                routes[idx] = best_route
                new_max = max(route_distance(r) for r in routes)
                if new_max < best_max - 1e-12:
                    report_best_vrp(routes)
                improved = True

    return best_routes if best_routes is not None else routes