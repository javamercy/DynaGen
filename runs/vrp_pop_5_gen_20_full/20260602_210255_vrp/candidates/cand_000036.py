import numpy as np
from copy import deepcopy

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # Initialization: all routes start and end at depot
    routes = [[0, 0] for _ in range(truck_count)]
    route_dist = [0.0] * truck_count
    unvisited = set(range(1, n))

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # Regret insertion construction
    while unvisited:
        best_customer = None
        best_regret = -1e100
        best_route_idx = None
        best_pos = None
        for cust in unvisited:
            costs = []
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                    costs.append((inc, r_idx, pos))
            costs.sort(key=lambda x: x[0])
            best_cost = costs[0][0]
            second_best_cost = costs[1][0] if len(costs) > 1 else best_cost
            regret = second_best_cost - best_cost
            if regret > best_regret:
                best_regret = regret
                best_customer = cust
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]
        # Insert best customer
        routes[best_route_idx].insert(best_pos, best_customer)
        route_dist[best_route_idx] = route_distance(routes[best_route_idx])
        unvisited.remove(best_customer)

    # Best solution so far
    best_routes = deepcopy(routes)
    best_max_dist = max(route_dist)
    report_best_vrp(best_routes)

    # Improvement: deterministic best-improvement with occasional worsening
    max_iter = n * truck_count
    for _ in range(max_iter):
        best_move = None
        best_new_max = 1e100
        best_total_inc = 1e100
        # Evaluate all possible relocate moves (customer to different route)
        for src_idx, src_route in enumerate(routes):
            if len(src_route) <= 2:
                continue
            for pos_in_src in range(1, len(src_route)-1):
                cust = src_route[pos_in_src]
                # Compute source route without this customer
                new_src = src_route[:pos_in_src] + src_route[pos_in_src+1:]
                new_src_dist = route_distance(new_src)
                for dst_idx, dst_route in enumerate(routes):
                    if dst_idx == src_idx:
                        continue
                    for p in range(1, len(dst_route)):
                        new_dst = dst_route[:p] + [cust] + dst_route[p:]
                        new_dst_dist = route_distance(new_dst)
                        # New max distance
                        new_max = new_src_dist
                        if new_dst_dist > new_max:
                            new_max = new_dst_dist
                        for i, d in enumerate(route_dist):
                            if i not in (src_idx, dst_idx):
                                if d > new_max:
                                    new_max = d
                        # Total distance increase (for tie-breaking)
                        total_inc = (new_src_dist + new_dst_dist) - (route_dist[src_idx] + route_dist[dst_idx])
                        if new_max < best_new_max - 1e-9 or (abs(new_max - best_new_max) < 1e-9 and total_inc < best_total_inc):
                            best_new_max = new_max
                            best_total_inc = total_inc
                            best_move = (src_idx, pos_in_src, dst_idx, p, cust, new_src, new_dst, new_src_dist, new_dst_dist)
        if best_move is None:
            break
        # Apply move
        src_idx, pos_in_src, dst_idx, p, cust, new_src, new_dst, nsd, ndd = best_move
        routes[src_idx] = new_src
        routes[dst_idx] = new_dst
        route_dist[src_idx] = nsd
        route_dist[dst_idx] = ndd
        if best_new_max < best_max_dist - 1e-9:
            best_max_dist = best_new_max
            best_routes = deepcopy(routes)
            report_best_vrp(best_routes)
        # Continue even if worsening

    return best_routes