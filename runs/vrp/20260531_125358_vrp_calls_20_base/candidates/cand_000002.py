import numpy as np
import itertools

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    unvisited = list(range(1, n))
    # sort customers by distance from depot descending
    unvisited.sort(key=lambda c: distance_matrix[0, c], reverse=True)
    
    def route_distance(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist
    
    # insertion construction
    for cust in unvisited:
        best_route = -1
        best_pos = -1
        best_new_dist = float('inf')
        for r_idx, route in enumerate(routes):
            cur_dist = route_distance(route)
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance(new_route)
                increase = new_dist - cur_dist
                # we want to minimize the new distance of that route
                if new_dist < best_new_dist or (new_dist == best_new_dist and r_idx < best_route):
                    best_new_dist = new_dist
                    best_route = r_idx
                    best_pos = pos
        # insert
        route = routes[best_route]
        routes[best_route] = route[:best_pos] + [cust] + route[best_pos:]
    
    # call report for initial solution
    report_best_vrp(routes)
    
    # improvement: 2-opt within each route
    for r_idx in range(truck_count):
        route = routes[r_idx]
        improved = True
        max_iter = n  # bounded
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            best_i = -1
            best_j = -1
            best_dist = route_distance(route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - 1e-9:
                        best_dist = new_dist
                        best_i, best_j = i, j
                        improved = True
            if improved:
                route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                routes[r_idx] = route
    
    # relocate between routes to reduce max distance
    max_dist = max(route_distance(r) for r in routes)
    improved = True
    max_iter = n * truck_count
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        current_max = max(route_distance(r) for r in routes)
        for cust in range(1, n):
            src_route_idx = None
            cust_pos_in_src = None
            for idx, route in enumerate(routes):
                if cust in route:
                    src_route_idx = idx
                    cust_pos_in_src = route.index(cust)
                    break
            if src_route_idx is None:
                continue
            src_route = routes[src_route_idx]
            # remove customer from src
            new_src = src_route[:cust_pos_in_src] + src_route[cust_pos_in_src+1:]
            src_dist = route_distance(new_src)
            for dst_route_idx in range(truck_count):
                if dst_route_idx == src_route_idx:
                    continue
                dst_route = routes[dst_route_idx]
                # try inserting at each position
                for pos in range(1, len(dst_route)):
                    new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                    dst_dist = route_distance(new_dst)
                    new_max = max(src_dist, dst_dist)
                    # also consider other routes unchanged
                    for other_idx in range(truck_count):
                        if other_idx != src_route_idx and other_idx != dst_route_idx:
                            new_max = max(new_max, route_distance(routes[other_idx]))
                    if new_max < current_max - 1e-9:
                        # accept move
                        routes[src_route_idx] = new_src
                        routes[dst_route_idx] = new_dst
                        current_max = new_max
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            report_best_vrp(routes)
    
    return routes