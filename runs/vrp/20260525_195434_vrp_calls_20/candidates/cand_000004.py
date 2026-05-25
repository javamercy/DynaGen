import numpy as np
import math
from collections import defaultdict

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    routes = [[depot, depot] for _ in range(truck_count)]
    unassigned = set(range(1, n))

    def route_distance(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist

    # 2-regret construction
    while unassigned:
        cust_info = []
        for cust in unassigned:
            deltas = []
            positions = []
            for ridx, route in enumerate(routes):
                best_pos = -1
                best_delta = float('inf')
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = pos
                deltas.append(best_delta)
                positions.append(best_pos)
            sorted_deltas = sorted(enumerate(deltas), key=lambda x: x[1])
            best_ridx, best_delta = sorted_deltas[0]
            second_best_delta = sorted_deltas[1][1] if len(sorted_deltas) > 1 else best_delta
            regret = second_best_delta - best_delta
            # Tie-breaking: regret descending, then best_delta ascending, then customer index
            cust_info.append((regret, best_delta, cust, best_ridx, positions[best_ridx]))
        # Sort by (-regret, best_delta, cust)
        cust_info.sort(key=lambda x: (-x[0], x[1], x[2]))
        _, _, cust, ridx, pos = cust_info[0]
        route = routes[ridx]
        route.insert(pos, cust)
        unassigned.remove(cust)

    # Initial best
    route_dists = [route_distance(r) for r in routes]
    best_routes = [r[:] for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)

    # Intra-route 2-opt on each route
    for idx in range(truck_count):
        route = routes[idx]
        if len(route) <= 3:
            continue
        improved = True
        max_iter = len(route) * len(route)
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                    new = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                    if new < old:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        iter_count += 1
                        break
                if improved:
                    break
        routes[idx] = route

    # Update after 2-opt
    route_dists = [route_distance(r) for r in routes]
    current_max = max(route_dists)
    if current_max < best_max:
        best_routes = [r[:] for r in routes]
        best_max = current_max
        report_best_vrp(best_routes)

    # Inter-route balancing: move customers from longest route to reduce max distance
    max_outer_iter = n
    outer_iter = 0
    improved = True
    while improved and outer_iter < max_outer_iter:
        improved = False
        outer_iter += 1
        longest_idx = max(range(truck_count), key=lambda i: route_dists[i])
        longest_route = routes[longest_idx]
        if len(longest_route) <= 2:
            break
        # Try moving each customer from longest to another route
        for cust in longest_route[1:-1]:
            # Remove customer from longest temporarily
            new_longest = [c for c in longest_route if c != cust]
            new_longest_dist = route_distance(new_longest)
            for other_idx in range(truck_count):
                if other_idx == longest_idx:
                    continue
                other_route = routes[other_idx]
                best_new_other = None
                best_new_other_dist = float('inf')
                best_pos = -1
                for pos in range(1, len(other_route)):
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_other_dist = route_distance(new_other)
                    if new_other_dist < best_new_other_dist:
                        best_new_other_dist = new_other_dist
                        best_new_other = new_other
                        best_pos = pos
                # Compute max distance if move performed
                new_max = max(new_longest_dist, best_new_other_dist)
                # Compare with current max (including other routes unchanged)
                temp_max = new_max
                for k in range(truck_count):
                    if k == longest_idx or k == other_idx:
                        continue
                    if route_dists[k] > temp_max:
                        temp_max = route_dists[k]
                if temp_max < current_max:
                    # Apply move
                    routes[longest_idx] = new_longest
                    routes[other_idx] = best_new_other
                    route_dists[longest_idx] = new_longest_dist
                    route_dists[other_idx] = best_new_other_dist
                    current_max = temp_max
                    improved = True
                    break
            if improved:
                break
        if improved:
            if current_max < best_max:
                best_routes = [r[:] for r in routes]
                best_max = current_max
                report_best_vrp(best_routes)

    return best_routes