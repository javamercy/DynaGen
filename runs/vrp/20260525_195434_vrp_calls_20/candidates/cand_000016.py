import numpy as np
import math
import random
import heapq
import itertools
from collections import defaultdict

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    
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

    # Construction: 2-regret (deterministic)
    routes = [[depot, depot] for _ in range(truck_count)]
    unassigned = set(range(1, n))
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
            best_delta = sorted_deltas[0][1]
            second_best_delta = sorted_deltas[1][1] if len(sorted_deltas) > 1 else best_delta
            regret = second_best_delta - best_delta
            best_ridx = sorted_deltas[0][0]
            cust_info.append((-regret, best_delta, cust, best_ridx, positions[best_ridx]))
        # primary: highest regret (negative sign), tie-break: smallest best_delta, then cust
        cust_info.sort(key=lambda x: (x[0], x[1], x[2]))
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

    # Improvement: steepest descent focusing on longest route
    max_outer = min(10, n)
    for _ in range(max_outer):
        improved_any = False
        while True:
            current_max = max(route_dists)
            max_idx = route_dists.index(current_max)
            best_new_max = current_max
            best_move = None
            route = routes[max_idx]

            # Inter-route relocate
            for i in range(1, len(route)-1):
                cust = route[i]
                new_route = route[:i] + route[i+1:]
                new_dist = route_distance(new_route)
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other = routes[other_idx]
                    pos, _ = best_insertion(other, cust)
                    new_other = other[:pos] + [cust] + other[pos:]
                    new_other_dist = route_distance(new_other)
                    cand_max = max(new_dist, new_other_dist)
                    if cand_max < best_new_max:
                        best_new_max = cand_max
                        best_move = ('relocate', max_idx, i, other_idx, pos)

            # Inter-route swap
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other = routes[other_idx]
                for i in range(1, len(route)-1):
                    for j in range(1, len(other)-1):
                        new_route = route[:i] + [other[j]] + route[i+1:]
                        new_other = other[:j] + [route[i]] + other[j+1:]
                        new_dist = route_distance(new_route)
                        new_other_dist = route_distance(new_other)
                        cand_max = max(new_dist, new_other_dist)
                        if cand_max < best_new_max:
                            best_new_max = cand_max
                            best_move = ('swap', max_idx, i, other_idx, j)

            # Intra-route 2-opt
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    other_max = max(d for idx, d in enumerate(route_dists) if idx != max_idx)
                    cand_max = max(new_dist, other_max)
                    if cand_max < best_new_max:
                        best_new_max = cand_max
                        best_move = ('2opt', max_idx, i, j)

            # Intra-route Or-opt (segment lengths 1-3)
            for seg_len in [1, 2, 3]:
                for start in range(1, len(route)-seg_len):
                    segment = route[start:start+seg_len]
                    temp_route = route[:start] + route[start+seg_len:]
                    for insert_pos in range(1, len(temp_route)):
                        new_route = temp_route[:insert_pos] + segment + temp_route[insert_pos:]
                        new_dist = route_distance(new_route)
                        other_max = max(d for idx, d in enumerate(route_dists) if idx != max_idx)
                        cand_max = max(new_dist, other_max)
                        if cand_max < best_new_max:
                            best_new_max = cand_max
                            best_move = ('oropt', max_idx, start, seg_len, insert_pos)

            if best_move is not None:
                # Apply move
                if best_move[0] == 'relocate':
                    _, max_idx, i, other_idx, pos = best_move
                    cust = routes[max_idx][i]
                    routes[max_idx] = routes[max_idx][:i] + routes[max_idx][i+1:]
                    routes[other_idx] = routes[other_idx][:pos] + [cust] + routes[other_idx][pos:]
                elif best_move[0] == 'swap':
                    _, max_idx, i, other_idx, j = best_move
                    cust_i = routes[max_idx][i]
                    cust_j = routes[other_idx][j]
                    routes[max_idx] = routes[max_idx][:i] + [cust_j] + routes[max_idx][i+1:]
                    routes[other_idx] = routes[other_idx][:j] + [cust_i] + routes[other_idx][j+1:]
                elif best_move[0] == '2opt':
                    _, max_idx, i, j = best_move
                    route = routes[max_idx]
                    routes[max_idx] = route[:i] + route[i:j+1][::-1] + route[j+1:]
                elif best_move[0] == 'oropt':
                    _, max_idx, start, seg_len, insert_pos = best_move
                    route = routes[max_idx]
                    segment = route[start:start+seg_len]
                    temp_route = route[:start] + route[start+seg_len:]
                    routes[max_idx] = temp_route[:insert_pos] + segment + temp_route[insert_pos:]

                route_dists = [route_distance(r) for r in routes]
                new_max = max(route_dists)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [r[:] for r in routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                improved_any = True
            else:
                break
        if not improved_any:
            break

    while len(best_routes) < truck_count:
        best_routes.append([depot, depot])
    return best_routes