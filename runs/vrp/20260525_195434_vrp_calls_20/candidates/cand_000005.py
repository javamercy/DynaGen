import numpy as np
import math
import random
import heapq
import itertools
from collections import defaultdict

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    routes = [[depot, depot] for _ in range(truck_count)]
    unassigned = set(range(1, n))

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(route, cust):
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

    # 3-regret construction
    while unassigned:
        cust_info = []
        for cust in unassigned:
            deltas = []
            positions = []
            for ridx, route in enumerate(routes):
                pos, delta = best_insertion(route, cust)
                deltas.append(delta)
                positions.append(pos)
            sorted_deltas = sorted(enumerate(deltas), key=lambda x: x[1])
            # collect up to 3 best deltas
            top_deltas = [d for _, d in sorted_deltas[:3]]
            if len(top_deltas) < 3:
                regret = sum(top_deltas) - top_deltas[0]
            else:
                regret = top_deltas[2] - top_deltas[0]
            best_ridx = sorted_deltas[0][0]
            best_delta = sorted_deltas[0][1]
            cust_info.append((regret, best_delta, cust, best_ridx, positions[best_ridx]))
        cust_info.sort(key=lambda x: (-x[0], x[1], x[2]))
        _, _, cust, ridx, pos = cust_info[0]
        routes[ridx].insert(pos, cust)
        unassigned.remove(cust)

    route_dists = [route_distance(r) for r in routes]
    best_routes = [r[:] for r in routes]
    best_max = max(route_dists)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    # Local search: iterate until no improvement
    improved = True
    while improved:
        improved = False
        current_max = max(route_dists)
        # try to improve longest routes first
        max_indices = [i for i, d in enumerate(route_dists) if d == current_max]
        for max_idx in max_indices:
            route = routes[max_idx]
            found = False
            # inter-route relocate from longest to others
            for i in range(1, len(route)-1):
                cust = route[i]
                new_route = route[:i] + route[i+1:]
                new_dist = route_distance(new_route)
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for pos in range(1, len(other_route)):
                        new_other = other_route[:pos] + [cust] + other_route[pos:]
                        new_other_dist = route_distance(new_other)
                        candidate_max = max(new_dist, new_other_dist)
                        if candidate_max < current_max:
                            routes[max_idx] = new_route
                            routes[other_idx] = new_other
                            route_dists[max_idx] = new_dist
                            route_dists[other_idx] = new_other_dist
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                continue
            # inter-route swap between longest and others
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for i in range(1, len(route)-1):
                    for j in range(1, len(other_route)-1):
                        # swap customers at positions i and j
                        new_route = route[:i] + [other_route[j]] + route[i+1:]
                        new_other = other_route[:j] + [route[i]] + other_route[j+1:]
                        new_dist = route_distance(new_route)
                        new_other_dist = route_distance(new_other)
                        candidate_max = max(new_dist, new_other_dist)
                        if candidate_max < current_max:
                            routes[max_idx] = new_route
                            routes[other_idx] = new_other
                            route_dists[max_idx] = new_dist
                            route_dists[other_idx] = new_other_dist
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                continue
            # intra-route 2-opt on longest
            best_route = route[:]
            best_dist = route_dists[max_idx]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist:
                        best_route = new_route
                        best_dist = new_dist
            if best_dist < route_dists[max_idx]:
                routes[max_idx] = best_route
                route_dists[max_idx] = best_dist
                improved = True
                found = True
                # update current_max? We'll just continue
        if improved:
            new_max = max(route_dists)
            if new_max < best_max:
                best_max = new_max
                best_routes = [r[:] for r in routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
    while len(best_routes) < truck_count:
        best_routes.append([depot, depot])
    return best_routes