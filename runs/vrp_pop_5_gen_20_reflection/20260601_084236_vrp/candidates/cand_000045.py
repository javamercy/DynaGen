import numpy as np
import random
import heapq

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    if truck_count >= n:
        routes = []
        for i in range(1, n):
            routes.append([0, i, 0])
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        best_route = route[:]
        best_dist = route_distance(route)
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new_route)
                    if d < best_dist - 1e-12:
                        best_dist = d
                        best_route = new_route
                        improved = True
                if improved:
                    break
            route = best_route
        return route

    def construct_initial_minmax(seed):
        random.seed(seed)
        # start with empty routes
        routes = [[0,0] for _ in range(truck_count)]
        unvisited = set(customers)
        # insert customers one by one into the route that minimizes the maximum route distance after insertion
        for cust in list(unvisited):
            best_max = float('inf')
            best_route_idx = 0
            best_pos = 1
            for t in range(truck_count):
                route = routes[t]
                # consider all insertion positions
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_distance(new_route)
                    # compute new max distance among all routes
                    other_max = 0.0
                    for t2 in range(truck_count):
                        if t2 == t:
                            d = new_dist
                        else:
                            d = route_distance(routes[t2])
                        if d > other_max:
                            other_max = d
                    if other_max < best_max - 1e-12:
                        best_max = other_max
                        best_route_idx = t
                        best_pos = pos
            # insert at best position
            route = routes[best_route_idx]
            routes[best_route_idx] = route[:best_pos] + [cust] + route[best_pos:]
            unvisited.remove(cust)
        # improve each route with 2-opt
        for t in range(truck_count):
            routes[t] = two_opt(routes[t])
        return routes

    def local_search(routes, best_routes, best_max):
        # intra-route 2-opt
        for t in range(truck_count):
            new_route = two_opt(routes[t])
            if route_distance(new_route) < route_distance(routes[t]) - 1e-12:
                routes[t] = new_route
                cur_max = max_distance(routes)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)

        # inter-route relocate and swap (1 iteration, limited to avoid huge loops)
        # try all pairs of routes
        for t1 in range(truck_count):
            for t2 in range(truck_count):
                if t1 == t2:
                    continue
                r1 = routes[t1]
                r2 = routes[t2]
                if len(r1) <= 2 or len(r2) <= 2:
                    continue
                # relocate moves: move a customer from r1 to r2
                for i in range(1, len(r1)-1):
                    cust = r1[i]
                    new_r1 = r1[:i] + r1[i+1:]
                    for j in range(1, len(r2)):
                        new_r2 = r2[:j] + [cust] + r2[j:]
                        new_r1_opt = two_opt(new_r1)
                        new_r2_opt = two_opt(new_r2)
                        d1 = route_distance(new_r1_opt)
                        d2 = route_distance(new_r2_opt)
                        other_max = 0.0
                        for idx, r in enumerate(routes):
                            if idx not in (t1, t2):
                                d = route_distance(r)
                                if d > other_max:
                                    other_max = d
                        cand_max = max(d1, d2, other_max)
                        if cand_max < best_max - 1e-12:
                            routes[t1] = new_r1_opt
                            routes[t2] = new_r2_opt
                            cur_max = max_distance(routes)
                            if cur_max < best_max - 1e-12:
                                best_max = cur_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                            # restart search from beginning? For simplicity, just continue.
                # swap moves
                for i in range(1, len(r1)-1):
                    for j in range(1, len(r2)-1):
                        cust1 = r1[i]
                        cust2 = r2[j]
                        new_r1 = r1[:i] + [cust2] + r1[i+1:]
                        new_r2 = r2[:j] + [cust1] + r2[j+1:]
                        new_r1_opt = two_opt(new_r1)
                        new_r2_opt = two_opt(new_r2)
                        d1 = route_distance(new_r1_opt)
                        d2 = route_distance(new_r2_opt)
                        other_max = 0.0
                        for idx, r in enumerate(routes):
                            if idx not in (t1, t2):
                                d = route_distance(r)
                                if d > other_max:
                                    other_max = d
                        cand_max = max(d1, d2, other_max)
                        if cand_max < best_max - 1e-12:
                            routes[t1] = new_r1_opt
                            routes[t2] = new_r2_opt
                            cur_max = max_distance(routes)
                            if cur_max < best_max - 1e-12:
                                best_max = cur_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
        return routes, best_routes, best_max, False

    best_routes = None
    best_max = float('inf')
    max_restarts = max(truck_count, 10)
    for restart in range(max_restarts):
        routes = construct_initial_minmax(restart)
        current_max = max_distance(routes)
        if current_max < best_max - 1e-12:
            best_routes = [r[:] for r in routes]
            best_max = current_max
            report_best_vrp(best_routes)
        # run local search a bounded number of times
        for iteration in range(n * truck_count):
            routes, best_routes, best_max, _ = local_search(routes, best_routes, best_max)
    # ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes