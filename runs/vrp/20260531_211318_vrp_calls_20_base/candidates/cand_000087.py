import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    # Initial construction: farthest customer first, insert into truck with smallest current route distance (balance), at cheapest position
    customers = sorted(range(1, n), key=lambda c: -dist[0][c])
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for cust in customers:
        # find truck with smallest current distance (tie: smallest index)
        min_dist_truck = min(range(truck_count), key=lambda t: route_dists[t])
        route = routes[min_dist_truck]
        best_pos = None
        best_increase = float('inf')
        for pos in range(1, len(route)):
            prev = route[pos-1]
            nxt = route[pos]
            increase = dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]
            if increase < best_increase - 1e-12:
                best_increase = increase
                best_pos = pos
            elif abs(increase - best_increase) < 1e-12 and pos < best_pos:
                best_pos = pos
        # insert
        routes[min_dist_truck] = route[:best_pos] + [cust] + route[best_pos:]
        route_dists[min_dist_truck] += best_increase

    best_routes = [list(r) for r in routes]
    best_dists = list(route_dists)
    best_max = max(best_dists)
    report_best_vrp(best_routes)

    # Balancing moves: move customers from longest route to shortest
    max_iter = 100
    for it in range(max_iter):
        improved = False
        # identify longest and shortest routes
        max_idx = max(range(truck_count), key=lambda t: route_dists[t])
        min_idx = min(range(truck_count), key=lambda t: route_dists[t])
        if max_idx == min_idx:
            break  # all equal, no balancing possible
        max_route = routes[max_idx]
        min_route = routes[min_idx]
        best_move = None
        best_new_max = best_max
        for pos in range(1, len(max_route)-1):
            cust = max_route[pos]
            # remove from max
            new_max_route = max_route[:pos] + max_route[pos+1:]
            new_max_dist = route_distance(new_max_route)
            # insert into min at best position
            for pos2 in range(1, len(min_route)):
                prev = min_route[pos2-1]
                nxt = min_route[pos2]
                increase = dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]
                new_min_dist = route_dists[min_idx] + increase
                other_dists = [route_dists[t] for t in range(truck_count) if t not in (max_idx, min_idx)]
                new_max_local = max([new_max_dist, new_min_dist] + other_dists)
                if new_max_local < best_new_max - 1e-12:
                    best_new_max = new_max_local
                    best_move = (max_idx, pos, min_idx, pos2, cust, new_max_route, new_max_dist, increase, new_min_dist)
        if best_move is not None:
            max_idx, pos, min_idx, pos2, cust, new_max_route, new_max_dist, increase, new_min_dist = best_move
            routes[max_idx] = new_max_route
            routes[min_idx] = routes[min_idx][:pos2] + [cust] + routes[min_idx][pos2:]
            route_dists[max_idx] = new_max_dist
            route_dists[min_idx] = new_min_dist
            current_max = max(route_dists)
            if current_max < best_max - 1e-12:
                best_max = current_max
                best_routes = [list(r) for r in routes]
                best_dists = list(route_dists)
                report_best_vrp(best_routes)
            improved = True
        if not improved:
            break  # no further improvement from balancing

    # Intra-route 2-opt on best solution
    for t in range(truck_count):
        route = best_routes[t]
        if len(route) <= 3:
            continue
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_dist = route_distance(new_route)
                if new_dist < best_dists[t] - 1e-12:
                    new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                    new_total = sum(best_dists) - best_dists[t] + new_dist
                    if new_max < best_max - 1e-12 or (abs(new_max - best_max) < 1e-12 and new_total < sum(best_dists)):
                        best_routes[t] = new_route
                        best_dists[t] = new_dist
                        best_max = new_max
                        report_best_vrp(best_routes)

    return best_routes