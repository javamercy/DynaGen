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
        routes = [[0, i, 0] for i in range(1, n)]
        for _ in range(truck_count - (n - 1)):
            routes.append([0, 0])
        return routes

    def route_distance(route):
        if len(route) == 2:
            return distance_matrix[0, 0]
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def build_giant_tour(seed):
        random.seed(seed)
        unvisited = list(range(1, n))
        random.shuffle(unvisited)
        tour = [0]
        current = 0
        while unvisited:
            # nearest neighbor with random tie-breaking
            next_c = min(unvisited, key=lambda c: distance_matrix[current, c])
            # if multiple have same distance, pick randomly among them
            candidates = [c for c in unvisited if distance_matrix[current, c] == distance_matrix[current, next_c]]
            next_c = random.choice(candidates)
            tour.append(next_c)
            unvisited.remove(next_c)
            current = next_c
        tour.append(0)
        return tour

    def greedy_split(giant_tour, threshold):
        customers = giant_tour[1:-1]
        routes = []
        current_route = [0]
        current_dist = 0.0
        for c in customers:
            if len(current_route) == 1:
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
                    current_route.append(0)
                    routes.append(current_route)
                    if distance_matrix[0, c] + distance_matrix[c, 0] > threshold:
                        return None, False
                    current_route = [0, c]
                    current_dist = distance_matrix[0, c]
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
        if len(routes) > truck_count:
            return None, False
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes, True

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

    def balance_routes(routes, best_max):
        max_iters = n * truck_count * 2
        improved = True
        while improved:
            improved = False
            lengths = [(route_distance(r), idx) for idx, r in enumerate(routes)]
            lengths.sort(key=lambda x: x[0])
            if lengths[0][0] == lengths[-1][0]:
                break
            longest_idx = lengths[-1][1]
            shortest_idx = lengths[0][1]
            longest_route = routes[longest_idx]
            shortest_route = routes[shortest_idx]
            custs_long = longest_route[1:-1]
            for cust in custs_long:
                new_long = [0] + [c for c in longest_route[1:-1] if c != cust] + [0]
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
                new_max = max(route_distance(new_long), route_distance(new_short),
                              max(route_distance(r) for idx, r in enumerate(routes) if idx not in [longest_idx, shortest_idx]))
                if new_max < best_max:
                    routes[longest_idx] = two_opt(new_long)
                    routes[shortest_idx] = two_opt(new_short)
                    best_max = new_max
                    improved = True
                    report_best_vrp(routes)
                    break
            if improved:
                continue
            custs_short = shortest_route[1:-1]
            for cust_a in custs_long:
                for cust_b in custs_short:
                    new_long = [0] + [c for c in longest_route[1:-1] if c != cust_a] + [cust_b] + [0]
                    new_short = [0] + [c for c in shortest_route[1:-1] if c != cust_b] + [cust_a] + [0]
                    new_long = two_opt(new_long)
                    new_short = two_opt(new_short)
                    new_max = max(route_distance(new_long), route_distance(new_short),
                                  max(route_distance(r) for idx, r in enumerate(routes) if idx not in [longest_idx, shortest_idx]))
                    if new_max < best_max:
                        routes[longest_idx] = new_long
                        routes[shortest_idx] = new_short
                        best_max = new_max
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break
        return routes, best_max

    best_routes = None
    best_max = float('inf')
    num_restarts = max(1, n * truck_count)
    for restart in range(num_restarts):
        giant_tour = build_giant_tour(restart)
        total_giant = route_distance(giant_tour)
        low = 0.0
        high = total_giant
        local_routes = None
        local_max = total_giant
        for _ in range(50):
            mid = (low + high) / 2.0
            routes, feasible = greedy_split(giant_tour, mid)
            if feasible:
                high = mid
                if mid < local_max:
                    local_max = mid
                    local_routes = [r[:] for r in routes]
            else:
                low = mid
        if local_routes is None:
            local_routes, _ = greedy_split(giant_tour, high)
            local_max = max_distance(local_routes)
        # apply 2-opt
        for idx in range(truck_count):
            if len(local_routes[idx]) > 2:
                local_routes[idx] = two_opt(local_routes[idx])
        local_max = max_distance(local_routes)
        # apply balancing
        local_routes, local_max = balance_routes(local_routes, local_max)
        if local_max < best_max:
            best_max = local_max
            best_routes = [r[:] for r in local_routes]
            try:
                report_best_vrp(best_routes)
            except NameError:
                pass
    return best_routes