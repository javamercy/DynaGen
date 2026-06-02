import numpy as np
from typing import List

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> List[List[int]]:
    n = distance_matrix.shape[0]
    # Construction
    routes = [[0] for _ in range(truck_count)]
    route_dist = [0.0] * truck_count
    customers = list(range(1, n))
    customers.sort(key=lambda c: distance_matrix[0, c])
    for cust in customers:
        best_new_max = float('inf')
        best_truck = -1
        best_pos = -1
        for t_idx in range(truck_count):
            route = routes[t_idx]
            for pos in range(1, len(route)):
                old_dist = route_dist[t_idx]
                new_dist = old_dist - distance_matrix[route[pos-1], route[pos]] + distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]]
                if truck_count == 1:
                    new_max = new_dist
                else:
                    other_max = max(route_dist[:t_idx] + route_dist[t_idx+1:])
                    new_max = max(new_dist, other_max)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_truck = t_idx
                    best_pos = pos
        routes[best_truck].insert(best_pos, cust)
        # recompute distance for that route
        d = 0.0
        r = routes[best_truck]
        for i in range(len(r)-1):
            d += distance_matrix[r[i], r[i+1]]
        route_dist[best_truck] = d
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist)
    report_best_vrp(best_routes)
    # Local search improvement
    max_iter = (n-1) * truck_count * 5
    for _ in range(max_iter):
        improved = False
        # Relocate moves
        for cust in range(1, n):
            src_truck = -1
            src_pos = -1
            for t_idx, route in enumerate(routes):
                if cust in route:
                    src_truck = t_idx
                    src_pos = route.index(cust)
                    break
            for tgt_truck in range(truck_count):
                if tgt_truck == src_truck:
                    continue
                tgt_route = routes[tgt_truck]
                for pos in range(1, len(tgt_route)):
                    # compute new distances
                    src_route = routes[src_truck]
                    old_src = route_dist[src_truck]
                    new_src = old_src - distance_matrix[src_route[src_pos-1], cust] - distance_matrix[cust, src_route[src_pos+1]] + distance_matrix[src_route[src_pos-1], src_route[src_pos+1]]
                    old_tgt = route_dist[tgt_truck]
                    new_tgt = old_tgt - distance_matrix[tgt_route[pos-1], tgt_route[pos]] + distance_matrix[tgt_route[pos-1], cust] + distance_matrix[cust, tgt_route[pos]]
                    other_max = 0.0
                    for idx, d in enumerate(route_dist):
                        if idx not in (src_truck, tgt_truck):
                            other_max = max(other_max, d)
                    new_max = max(new_src, new_tgt, other_max)
                    if new_max < best_max:
                        # execute move
                        del routes[src_truck][src_pos]
                        routes[tgt_truck].insert(pos, cust)
                        route_dist[src_truck] = new_src
                        route_dist[tgt_truck] = new_tgt
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
        # Swap moves
        for i in range(1, n):
            for j in range(i+1, n):
                r1 = -1; p1 = -1; r2 = -1; p2 = -1
                for t_idx, route in enumerate(routes):
                    if i in route:
                        r1 = t_idx; p1 = route.index(i)
                    if j in route:
                        r2 = t_idx; p2 = route.index(j)
                if r1 == -1 or r2 == -1 or r1 == r2:
                    continue
                route1 = routes[r1]
                route2 = routes[r2]
                # compute new distances
                old1 = route_dist[r1]
                new1 = old1 - distance_matrix[route1[p1-1], i] - distance_matrix[i, route1[p1+1]] + distance_matrix[route1[p1-1], j] + distance_matrix[j, route1[p1+1]]
                old2 = route_dist[r2]
                new2 = old2 - distance_matrix[route2[p2-1], j] - distance_matrix[j, route2[p2+1]] + distance_matrix[route2[p2-1], i] + distance_matrix[i, route2[p2+1]]
                other_max = 0.0
                for idx, d in enumerate(route_dist):
                    if idx not in (r1, r2):
                        other_max = max(other_max, d)
                new_max = max(new1, new2, other_max)
                if new_max < best_max:
                    # execute swap
                    del routes[r1][p1]
                    del routes[r2][p2]
                    routes[r1].insert(p1, j)
                    routes[r2].insert(p2, i)
                    route_dist[r1] = new1
                    route_dist[r2] = new2
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # Intra-route 2-opt
        for t_idx in range(truck_count):
            route = routes[t_idx]
            if len(route) < 4:
                continue
            for i in range(1, len(route)-2):
                for k in range(i+1, len(route)-1):
                    old_dist = route_dist[t_idx]
                    new_dist = old_dist - distance_matrix[route[i-1], route[i]] - distance_matrix[route[k], route[k+1]] + distance_matrix[route[i-1], route[k]] + distance_matrix[route[i], route[k+1]]
                    if new_dist < old_dist:
                        if truck_count == 1:
                            new_max = new_dist
                        else:
                            other_max = max(route_dist[:t_idx] + route_dist[t_idx+1:])
                            new_max = max(new_dist, other_max)
                        if new_max < best_max:
                            # execute 2-opt
                            route[i:k+1] = reversed(route[i:k+1])
                            route_dist[t_idx] = new_dist
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best_routes