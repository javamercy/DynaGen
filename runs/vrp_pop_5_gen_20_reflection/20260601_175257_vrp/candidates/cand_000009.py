import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))

    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i + 1]]
        return d

    # Construction: balanced regret insertion
    while unassigned:
        # compute average route distance
        dists = [route_dist(r) for r in routes]
        avg_dist = np.mean(dists) if dists else 0.0
        best_cust = None
        best_regret = -1.0
        best_cost_val = None
        best_route_idx = None
        for cust in unassigned:
            best_cost = None
            second_best_cost = None
            best_routes = []
            for r_idx, route in enumerate(routes):
                # find best insertion cost for cust in this route
                min_inc = float('inf')
                for pos in range(1, len(route)):
                    a = route[pos - 1]
                    b = route[pos]
                    inc = distance_matrix[a, cust] + distance_matrix[cust, b] - distance_matrix[a, b]
                    if inc < min_inc:
                        min_inc = inc
                # apply penalty based on route load
                route_after_dist = route_dist(route[:1] + [cust] + route[1:])  # approximate
                # actually compute exact after insertion
                # to avoid recomputation, we can estimate: new_dist = dists[r_idx] + min_inc
                new_dist = dists[r_idx] + min_inc
                if avg_dist > 0:
                    penalty = 1 + 0.5 * (new_dist / avg_dist - 1)
                else:
                    penalty = 1.0
                effective_inc = min_inc * penalty
                if best_cost is None or effective_inc < best_cost:
                    second_best_cost = best_cost
                    best_cost = effective_inc
                    best_routes = [r_idx]
                elif effective_inc == best_cost:
                    best_routes.append(r_idx)
                elif second_best_cost is None or effective_inc < second_best_cost:
                    second_best_cost = effective_inc
            if second_best_cost is None:
                regret = best_cost
            else:
                regret = second_best_cost - best_cost
            # tie-breaking: prefer larger regret, then smaller best_cost, then smaller customer index
            if (regret > best_regret) or (regret == best_regret and (best_cost_val is None or best_cost < best_cost_val)):
                best_regret = regret
                best_cust = cust
                best_cost_val = best_cost
                best_route_idx = best_routes[0]
            elif regret == best_regret and best_cost == best_cost_val and cust < best_cust:
                best_cust = cust
                best_route_idx = best_routes[0]
        # insert best_cust into best_route_idx at best position (using original inc)
        route = routes[best_route_idx]
        best_inc = float('inf')
        best_pos = None
        for pos in range(1, len(route)):
            a = route[pos - 1]
            b = route[pos]
            inc = distance_matrix[a, best_cust] + distance_matrix[best_cust, b] - distance_matrix[a, b]
            if inc < best_inc:
                best_inc = inc
                best_pos = pos
        route.insert(best_pos, best_cust)
        unassigned.remove(best_cust)

    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)

    # Improvement phase 1: intra-route 2-opt and inter-route relocate from longest
    improved = True
    max_iter_1 = n * 5
    iters = 0
    while improved and iters < max_iter_1:
        improved = False
        iters += 1
        # Intra-route 2-opt
        for r_idx, route in enumerate(routes):
            if len(route) <= 2:
                continue
            d = route_dist(route)
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_d = route_dist(new_route)
                    if new_d < d:
                        routes[r_idx] = new_route
                        d = new_d
                        improved = True
        # Inter-route relocate from longest route
        dists = [route_dist(r) for r in routes]
        current_max = max(dists)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        longest_idx = np.argmax(dists)
        longest_route = routes[longest_idx]
        for cust_idx in range(1, len(longest_route) - 1):
            cust = longest_route[cust_idx]
            for other_idx in range(truck_count):
                if other_idx == longest_idx:
                    continue
                other_route = routes[other_idx]
                best_inc = float('inf')
                best_pos = None
                for pos in range(1, len(other_route)):
                    a = other_route[pos - 1]
                    b = other_route[pos]
                    inc = distance_matrix[a, cust] + distance_matrix[cust, b] - distance_matrix[a, b]
                    if inc < best_inc:
                        best_inc = inc
                        best_pos = pos
                new_long = longest_route[:cust_idx] + longest_route[cust_idx+1:]
                new_other = other_route[:best_pos] + [cust] + other_route[best_pos:]
                new_dist_long = route_dist(new_long)
                new_dist_other = route_dist(new_other)
                other_dists = [dists[i] for i in range(truck_count) if i not in (longest_idx, other_idx)]
                new_max = max([new_dist_long, new_dist_other] + other_dists)
                if new_max < current_max:
                    routes[longest_idx] = new_long
                    routes[other_idx] = new_other
                    improved = True
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                    break
            if improved:
                break

    # Improvement phase 2: cross-exchange between pairs of routes
    max_iter_2 = n * 2
    for _ in range(max_iter_2):
        dists = [route_dist(r) for r in routes]
        current_max = max(dists)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        improved = False
        for i in range(truck_count):
            for j in range(i+1, truck_count):
                route_i = routes[i]
                route_j = routes[j]
                if len(route_i) <= 2 or len(route_j) <= 2:
                    continue
                # try all single customer swaps
                for ci in range(1, len(route_i)-1):
                    for cj in range(1, len(route_j)-1):
                        cust_i = route_i[ci]
                        cust_j = route_j[cj]
                        new_i = route_i[:ci] + route_i[ci+1:]
                        new_j = route_j[:cj] + route_j[cj+1:]
                        # insert cust_j into new_i at best position
                        best_pos_i = None
                        best_inc_i = float('inf')
                        for pos in range(1, len(new_i)):
                            a = new_i[pos-1]
                            b = new_i[pos]
                            inc = distance_matrix[a, cust_j] + distance_matrix[cust_j, b] - distance_matrix[a, b]
                            if inc < best_inc_i:
                                best_inc_i = inc
                                best_pos_i = pos
                        new_i2 = new_i[:best_pos_i] + [cust_j] + new_i[best_pos_i:]
                        # insert cust_i into new_j at best position
                        best_pos_j = None
                        best_inc_j = float('inf')
                        for pos in range(1, len(new_j)):
                            a = new_j[pos-1]
                            b = new_j[pos]
                            inc = distance_matrix[a, cust_i] + distance_matrix[cust_i, b] - distance_matrix[a, b]
                            if inc < best_inc_j:
                                best_inc_j = inc
                                best_pos_j = pos
                        new_j2 = new_j[:best_pos_j] + [cust_i] + new_j[best_pos_j:]
                        # evaluate
                        new_max = max(route_dist(new_i2), route_dist(new_j2))
                        if new_max < current_max:
                            routes[i] = new_i2
                            routes[j] = new_j2
                            improved = True
                            current_max = new_max
                            if current_max < best_max:
                                best_max = current_max
                                best_routes = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_routes