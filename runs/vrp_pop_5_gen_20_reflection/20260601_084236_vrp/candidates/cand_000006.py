import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count >= n - 1:
        routes = []
        for i in range(1, n):
            routes.append([0, i, 0])
        for _ in range(truck_count - (n - 1)):
            routes.append([0, 0])
        return routes

    # helper: route distance
    def route_distance(route):
        if len(route) == 2:
            return distance_matrix[0, 0]
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # helper: max distance over routes
    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    # build giant TSP tour (nearest neighbor, deterministic)
    def build_giant_tour():
        unvisited = set(range(1, n))
        tour = [0]
        current = 0
        while unvisited:
            next_c = min(unvisited, key=lambda c: (distance_matrix[current, c], c))
            tour.append(next_c)
            unvisited.remove(next_c)
            current = next_c
        tour.append(0)
        return tour

    giant_tour = build_giant_tour()
    total_giant = route_distance(giant_tour)

    # greedy split given threshold, returns (routes, feasible)
    def greedy_split(threshold):
        customers = giant_tour[1:-1]
        routes = []
        current_route = [0]
        current_dist = 0.0  # distance from depot to current last node (excludes return)
        for c in customers:
            if len(current_route) == 1:  # only depot
                # try to add c as first customer
                new_total = distance_matrix[0, c] + distance_matrix[c, 0]
                if new_total <= threshold:
                    current_route.append(c)
                    current_dist = distance_matrix[0, c]
                else:
                    return None, False
            else:
                last = current_route[-1]
                new_total = current_dist + distance_matrix[last, c] + distance_matrix[c, 0]
                if new_total <= threshold:
                    current_route.append(c)
                    current_dist += distance_matrix[last, c]
                else:
                    # close current route
                    current_route.append(0)
                    routes.append(current_route)
                    # start new route with c
                    if distance_matrix[0, c] + distance_matrix[c, 0] > threshold:
                        return None, False
                    current_route = [0, c]
                    current_dist = distance_matrix[0, c]
        # finish last route
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
        if len(routes) > truck_count:
            return None, False
        # pad with empty routes
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes, True

    # binary search for minimal max distance
    low = 0.0
    high = total_giant
    best_routes = None
    best_max = total_giant
    for _ in range(50):
        mid = (low + high) / 2.0
        routes, feasible = greedy_split(mid)
        if feasible:
            high = mid
            if mid < best_max:
                best_max = mid
                best_routes = [r[:] for r in routes]
        else:
            low = mid
    if best_routes is None:
        # fallback: split with high (should be feasible)
        best_routes, _ = greedy_split(high)
        best_max = max_distance(best_routes)

    # improvement: intra-route 2-opt
    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
        return route

    for idx in range(truck_count):
        if len(best_routes[idx]) > 2:
            best_routes[idx] = two_opt(best_routes[idx])
    best_max = max_distance(best_routes)
    # report best after 2-opt (optional, but not required)
    # we import report_best_vrp? It's assumed available; we'll call it if defined
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    # improvement: balancing moves (move/swap from longest to shortest)
    max_iters = n * truck_count * 2
    for _ in range(max_iters):
        improved = False
        # find longest and shortest routes (by distance)
        lengths = [(route_distance(r), idx) for idx, r in enumerate(best_routes)]
        lengths.sort(key=lambda x: x[0])
        if lengths[0][0] == lengths[-1][0]:
            break
        longest_idx = lengths[-1][1]
        shortest_idx = lengths[0][1]
        longest_route = best_routes[longest_idx]
        shortest_route = best_routes[shortest_idx]
        custs_long = longest_route[1:-1]
        # try moving a customer from longest to shortest
        for cust in custs_long:
            new_long = [0] + [c for c in longest_route[1:-1] if c != cust] + [0]
            # insert cust into shortest at best position
            best_pos = -1
            best_inc = float('inf')
            for pos in range(1, len(shortest_route)):
                prev = shortest_route[pos-1]
                nxt = shortest_route[pos]
                inc = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                if inc < best_inc:
                    best_inc = inc
                    best_pos = pos
            if best_pos == -1:
                continue
            new_short = shortest_route[:best_pos] + [cust] + shortest_route[best_pos:]
            # compute new max
            new_max = max(route_distance(new_long), route_distance(new_short), 
                          max(route_distance(r) for idx, r in enumerate(best_routes) if idx not in [longest_idx, shortest_idx]))
            if new_max < best_max:
                best_routes[longest_idx] = two_opt(new_long)
                best_routes[shortest_idx] = two_opt(new_short)
                best_max = new_max
                improved = True
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
                break
        if improved:
            continue
        # try swapping customers between longest and shortest
        custs_short = shortest_route[1:-1]
        for cust_a in custs_long:
            for cust_b in custs_short:
                # remove both
                new_long = [0] + [c for c in longest_route[1:-1] if c != cust_a] + [cust_b] + [0]
                new_short = [0] + [c for c in shortest_route[1:-1] if c != cust_b] + [cust_a] + [0]
                new_long = two_opt(new_long)
                new_short = two_opt(new_short)
                new_max = max(route_distance(new_long), route_distance(new_short),
                              max(route_distance(r) for idx, r in enumerate(best_routes) if idx not in [longest_idx, shortest_idx]))
                if new_max < best_max:
                    best_routes[longest_idx] = new_long
                    best_routes[shortest_idx] = new_short
                    best_max = new_max
                    improved = True
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                    break
            if improved:
                break
        if not improved:
            break

    return best_routes