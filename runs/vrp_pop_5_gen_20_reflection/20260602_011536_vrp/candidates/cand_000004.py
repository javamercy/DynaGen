import numpy as np
import math
from itertools import combinations, permutations, product

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # Initialize empty routes
    routes = [[0, 0] for _ in range(truck_count)]
    # Map from route idx to list of customers (order)
    route_custs = [[] for _ in range(truck_count)]

    def route_distance(route):
        if len(route) == 2:
            return 0.0
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist

    def get_best_insertion(customer, routes, route_custs):
        best_max = float('inf')
        best_route = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            # positions are from 1 to len(route)-1 (insert between nodes)
            for pos in range(1, len(route)):
                new_route = route[:pos] + [customer] + route[pos:]
                new_dist = route_distance(new_route)
                # compute new max over all routes
                new_max = new_dist
                for r2 in range(truck_count):
                    if r2 == r:
                        continue
                    d = route_distance(routes[r2])
                    if d > new_max:
                        new_max = d
                # also consider the route being empty? already handled
                if new_max < best_max or (new_max == best_max and (r < best_route or (r == best_route and pos < best_pos))):
                    best_max = new_max
                    best_route = r
                    best_pos = pos
        return best_route, best_pos, best_max

    # Insert customers sequentially
    for c in customers:
        r, pos, _ = get_best_insertion(c, routes, route_custs)
        # insert
        route = routes[r]
        new_route = route[:pos] + [c] + route[pos:]
        routes[r] = new_route
        route_custs[r].append(c)

    # Compute initial best max
    def compute_max():
        mx = 0.0
        for r in routes:
            d = route_distance(r)
            if d > mx:
                mx = d
        return mx

    best_max = compute_max()
    best_routes = [r[:] for r in routes]
    # report initial
    # report_best_vrp(best_routes)

    # Local search
    improved = True
    max_iter = 2 * n * truck_count  # finite bound
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        # Try all moves
        best_move = None
        best_new_max = best_max
        # --- Intra-route 2-opt ---
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 4:  # need at least 2 customers
                continue
            # nodes from index 1 to len-2 (excluding depot)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # reverse segment from i to j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    # check if actually different
                    if new_route == route:
                        continue
                    new_dist = route_distance(new_route)
                    # compute new max
                    new_max = new_dist
                    for r2 in range(truck_count):
                        if r2 == r:
                            continue
                        d = route_distance(routes[r2])
                        if d > new_max:
                            new_max = d
                    if new_max < best_new_max or (new_max == best_new_max and (r, i, j < best_move is not None)):
                        best_new_max = new_max
                        best_move = ('2opt', r, None, i, j)
        # --- Inter-route relocate ---
        for src in range(truck_count):
            route_src = routes[src]
            if len(route_src) <= 2:
                continue
            # customers in src are indices 1 to len-2
            for pos_cust in range(1, len(route_src)-1):
                cust = route_src[pos_cust]
                # remove customer
                route_src_new = route_src[:pos_cust] + route_src[pos_cust+1:]
                # try inserting into every other route
                for dst in range(truck_count):
                    if dst == src:
                        continue
                    route_dst = routes[dst]
                    for pos_ins in range(1, len(route_dst)):
                        new_route_dst = route_dst[:pos_ins] + [cust] + route_dst[pos_ins:]
                        new_dist_src = route_distance(route_src_new)
                        new_dist_dst = route_distance(new_route_dst)
                        new_max = max(new_dist_src, new_dist_dst)
                        for r2 in range(truck_count):
                            if r2 == src or r2 == dst:
                                continue
                            d = route_distance(routes[r2])
                            if d > new_max:
                                new_max = d
                        if new_max < best_new_max or (new_max == best_new_max and (src, pos_cust, dst, pos_ins < best_move is not None)):
                            best_new_max = new_max
                            best_move = ('relocate', src, dst, pos_cust, pos_ins)
        # --- Inter-route exchange ---
        for r1, r2 in combinations(range(truck_count), 2):
            route1 = routes[r1]
            route2 = routes[r2]
            if len(route1) <= 2 or len(route2) <= 2:
                continue
            # customers in each
            custs1 = list(range(1, len(route1)-1))
            custs2 = list(range(1, len(route2)-1))
            for p1 in custs1:
                cust1 = route1[p1]
                for p2 in custs2:
                    cust2 = route2[p2]
                    # swap
                    new_route1 = route1[:p1] + [cust2] + route1[p1+1:]
                    new_route2 = route2[:p2] + [cust1] + route2[p2+1:]
                    new_dist1 = route_distance(new_route1)
                    new_dist2 = route_distance(new_route2)
                    new_max = max(new_dist1, new_dist2)
                    for r3 in range(truck_count):
                        if r3 == r1 or r3 == r2:
                            continue
                        d = route_distance(routes[r3])
                        if d > new_max:
                            new_max = d
                    if new_max < best_new_max or (new_max == best_new_max and (r1, p1, r2, p2 < best_move is not None)):
                        best_new_max = new_max
                        best_move = ('exchange', r1, r2, p1, p2)

        if best_move is not None and best_new_max < best_max:
            # apply move
            if best_move[0] == '2opt':
                _, r, _, i, j = best_move
                route = routes[r]
                routes[r] = route[:i] + route[i:j+1][::-1] + route[j+1:]
            elif best_move[0] == 'relocate':
                _, src, dst, pos_cust, pos_ins = best_move
                cust = routes[src][pos_cust]
                routes[src] = routes[src][:pos_cust] + routes[src][pos_cust+1:]
                routes[dst] = routes[dst][:pos_ins] + [cust] + routes[dst][pos_ins:]
            else:  # exchange
                _, r1, r2, p1, p2 = best_move
                cust1 = routes[r1][p1]
                cust2 = routes[r2][p2]
                routes[r1][p1] = cust2
                routes[r2][p2] = cust1
            best_max = best_new_max
            improved = True
            # report new best
            # report_best_vrp([r[:] for r in routes])

    # After improvement, return best found
    return [r[:] for r in routes]