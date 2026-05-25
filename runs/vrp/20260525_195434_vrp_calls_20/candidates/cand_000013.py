import numpy as np
import random
import math
from collections import defaultdict

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0

    def route_distance(route):
        if len(route) <= 2:
            return distance_matrix[depot, depot] * 2  # should be 0 but safe
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insert(route, cust):
        best_pos = -1
        best_delta = float('inf')
        for pos in range(1, len(route)):
            prev = route[pos-1]
            nxt = route[pos]
            delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
        return best_pos, best_delta

    # Regret-2 construction (deterministic)
    routes = [[depot, depot] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    while unassigned:
        # Compute regret-2 for each customer
        best_cust = None
        best_regret = -float('inf')
        best_ridx = None
        best_pos = None
        best_delta = None
        for cust in unassigned:
            deltas = []
            positions = []
            for ridx, route in enumerate(routes):
                pos, delta = best_insert(route, cust)
                deltas.append(delta)
                positions.append(pos)
            sorted_deltas = sorted(deltas)
            if len(sorted_deltas) >= 2:
                regret = sorted_deltas[1] - sorted_deltas[0]
            else:
                regret = 0  # only one route, regret zero
            # Choose customer with max regret, tie-break by smallest best delta
            if regret > best_regret or (regret == best_regret and deltas[deltas.index(sorted_deltas[0])] < best_delta):
                best_regret = regret
                best_cust = cust
                best_ridx = deltas.index(sorted_deltas[0])
                best_pos = positions[best_ridx]
                best_delta = deltas[best_ridx]
        routes[best_ridx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)

    best_routes = [r[:] for r in routes]
    best_max = max(route_distance(r) for r in routes)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    # Local search: iterate over all pairs, O(n^2) bounded
    improved = True
    while improved:
        improved = False
        # Compute distances and max
        dists = [route_distance(r) for r in routes]
        current_max = max(dists)
        # Choose the longest route(s) to try moves
        max_indices = [i for i, d in enumerate(dists) if d == current_max]
        for max_idx in max_indices:
            route = routes[max_idx]
            # Inter-route relocate: move a customer from max route to another
            for i in range(1, len(route)-1):
                cust = route[i]
                new_src = route[:i] + route[i+1:]
                new_src_dist = route_distance(new_src)
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for j in range(1, len(other_route)):
                        new_other = other_route[:j] + [cust] + other_route[j:]
                        new_other_dist = route_distance(new_other)
                        cand_max = max(new_src_dist, new_other_dist)
                        if cand_max < current_max:
                            # Update
                            routes[max_idx] = new_src
                            routes[other_idx] = new_other
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-route swap: swap customers between max route and another
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for i in range(1, len(route)-1):
                    for j in range(1, len(other_route)-1):
                        new_src = route[:i] + [other_route[j]] + route[i+1:]
                        new_other = other_route[:j] + [route[i]] + other_route[j+1:]
                        new_src_dist = route_distance(new_src)
                        new_other_dist = route_distance(new_other)
                        cand_max = max(new_src_dist, new_other_dist)
                        if cand_max < current_max:
                            routes[max_idx] = new_src
                            routes[other_idx] = new_other
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Intra-route 2-opt on max route
            best_route = route[:]
            best_dist = route_distance(best_route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    # Check if it reduces max and not worsens other routes (others unchanged)
                    # Since only this route changes, if new_dist < current_max that's enough
                    if new_dist < current_max:
                        routes[max_idx] = new_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
        # After improvements, update best if better
        new_dists = [route_distance(r) for r in routes]
        new_max = max(new_dists)
        if new_max < best_max:
            best_max = new_max
            best_routes = [r[:] for r in routes]
            try:
                report_best_vrp(best_routes)
            except NameError:
                pass

    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([depot, depot])
    return best_routes