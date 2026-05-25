import numpy as np
import math
import random
import heapq
import itertools
import collections

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    
    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def compute_max_dist(routes):
        return max(route_distance(r) for r in routes)
    
    def construct():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = set(customers)
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_r = None
            best_pos = None
            best_new_max = None
            for cust in list(unassigned):
                best_cost = float('inf')
                second_best_cost = float('inf')
                best_r_temp = None
                best_p_temp = None
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        delta = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        new_dist = route_dists[r] + delta
                        new_max = new_dist
                        for rr in range(truck_count):
                            if rr != r:
                                new_max = max(new_max, route_dists[rr])
                        if new_max < best_cost:
                            second_best_cost = best_cost
                            best_cost = new_max
                            best_r_temp = r
                            best_p_temp = pos
                        elif new_max < second_best_cost:
                            second_best_cost = new_max
                regret = second_best_cost - best_cost
                if regret > best_regret or (regret == best_regret and cust < best_cust):
                    best_regret = regret
                    best_cust = cust
                    best_r = best_r_temp
                    best_pos = best_p_temp
                    best_new_max = best_cost
            routes[best_r].insert(best_pos, best_cust)
            route_dists[best_r] = route_distance(routes[best_r])
            unassigned.remove(best_cust)
        return routes, route_dists

    routes, route_dists = construct()
    best_routes = [r[:] for r in routes]
    best_max = compute_max_dist(routes)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass
    
    current_routes = [r[:] for r in routes]
    current_max = best_max
    route_dists = [route_distance(r) for r in current_routes]
    deviation = 0.1 * current_max
    max_iter = 200
    no_improve_iter = 0
    
    for it in range(max_iter):
        improved = False
        # best-improvement intra-2opt
        best_move = None
        best_new_max = current_max + deviation + 1
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+2, len(route)-1):
                    new_route = route[:i+1] + route[j:i:-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    cand_max = new_dist
                    for rr in range(truck_count):
                        if rr != r_idx:
                            cand_max = max(cand_max, route_dists[rr])
                    if cand_max < best_new_max:
                        best_new_max = cand_max
                        best_move = ('intra2opt', r_idx, i, j, new_route, new_dist)
        # best-improvement relocate
        for r_from in range(truck_count):
            if len(current_routes[r_from]) <= 2:
                continue
            for pos_from in range(1, len(current_routes[r_from])-1):
                cust = current_routes[r_from][pos_from]
                prev = current_routes[r_from][pos_from-1]
                nxt = current_routes[r_from][pos_from+1]
                delta_from = distance_matrix[prev, nxt] - distance_matrix[prev, cust] - distance_matrix[cust, nxt]
                new_from_dist = route_dists[r_from] + delta_from
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    route_to = current_routes[r_to]
                    for pos_to in range(1, len(route_to)):
                        prev_to = route_to[pos_to-1]
                        next_to = route_to[pos_to]
                        delta_to = distance_matrix[prev_to, cust] + distance_matrix[cust, next_to] - distance_matrix[prev_to, next_to]
                        new_to_dist = route_dists[r_to] + delta_to
                        cand_max = max(new_from_dist, new_to_dist)
                        for rr in range(truck_count):
                            if rr != r_from and rr != r_to:
                                cand_max = max(cand_max, route_dists[rr])
                        if cand_max < best_new_max:
                            best_new_max = cand_max
                            best_move = ('relocate', r_from, pos_from, r_to, pos_to, new_from_dist, new_to_dist, cust)
        # best-improvement swap
        for r1 in range(truck_count):
            if len(current_routes[r1]) <= 2:
                continue
            for pos1 in range(1, len(current_routes[r1])-1):
                cust1 = current_routes[r1][pos1]
                prev1 = current_routes[r1][pos1-1]
                next1 = current_routes[r1][pos1+1]
                for r2 in range(r1+1, truck_count):
                    if len(current_routes[r2]) <= 2:
                        continue
                    for pos2 in range(1, len(current_routes[r2])-1):
                        cust2 = current_routes[r2][pos2]
                        prev2 = current_routes[r2][pos2-1]
                        next2 = current_routes[r2][pos2+1]
                        delta1 = distance_matrix[prev1, cust2] + distance_matrix[cust2, next1] - distance_matrix[prev1, cust1] - distance_matrix[cust1, next1]
                        new_dist1 = route_dists[r1] + delta1
                        delta2 = distance_matrix[prev2, cust1] + distance_matrix[cust1, next2] - distance_matrix[prev2, cust2] - distance_matrix[cust2, next2]
                        new_dist2 = route_dists[r2] + delta2
                        cand_max = max(new_dist1, new_dist2)
                        for rr in range(truck_count):
                            if rr != r1 and rr != r2:
                                cand_max = max(cand_max, route_dists[rr])
                        if cand_max < best_new_max:
                            best_new_max = cand_max
                            best_move = ('swap', r1, pos1, r2, pos2, new_dist1, new_dist2, cust1, cust2)
        if best_move is not None and best_new_max <= current_max + deviation:
            if best_move[0] == 'intra2opt':
                _, r_idx, i, j, new_route, new_dist = best_move
                current_routes[r_idx] = new_route
                route_dists[r_idx] = new_dist
            elif best_move[0] == 'relocate':
                _, r_from, pos_from, r_to, pos_to, new_from_dist, new_to_dist, cust = best_move
                current_routes[r_from].pop(pos_from)
                current_routes[r_to].insert(pos_to, cust)
                route_dists[r_from] = new_from_dist
                route_dists[r_to] = new_to_dist
            else:  # swap
                _, r1, pos1, r2, pos2, new_dist1, new_dist2, cust1, cust2 = best_move
                current_routes[r1][pos1] = cust2
                current_routes[r2][pos2] = cust1
                route_dists[r1] = new_dist1
                route_dists[r2] = new_dist2
            current_max = compute_max_dist(current_routes)
            if current_max < best_max:
                best_max = current_max
                best_routes = [r[:] for r in current_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
                no_improve_iter = 0
            improved = True
        # diversification: 5% gap trigger
        if current_max > best_max * 1.05:
            max_dist = max(route_dists)
            longest_indices = [i for i, d in enumerate(route_dists) if d == max_dist]
            longest_idx = max(longest_indices, key=lambda i: len(current_routes[i]))
            if len(current_routes[longest_idx]) > 2:
                num_remove = max(1, (n-1) // 10)
                customers_in_route = [c for c in current_routes[longest_idx] if c != 0]
                customers_in_route.sort(key=lambda c: -distance_matrix[depot, c])
                to_remove = set(customers_in_route[:num_remove])
                new_route = [0]
                for c in current_routes[longest_idx][1:-1]:
                    if c not in to_remove:
                        new_route.append(c)
                new_route.append(0)
                current_routes[longest_idx] = new_route
                route_dists[longest_idx] = route_distance(new_route)
                unassigned = [c for c in customers_in_route if c in to_remove]
                while unassigned:
                    best_cust = None
                    best_regret = -1.0
                    best_r = None
                    best_pos = None
                    best_new_max = None
                    for cust in unassigned:
                        best_cost = float('inf')
                        second_best_cost = float('inf')
                        best_r_temp = None
                        best_p_temp = None
                        for r in range(truck_count):
                            route = current_routes[r]
                            for pos in range(1, len(route)):
                                delta = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                                new_dist = route_dists[r] + delta
                                new_max = new_dist
                                for rr in range(truck_count):
                                    if rr != r:
                                        new_max = max(new_max, route_dists[rr])
                                if new_max < best_cost:
                                    second_best_cost = best_cost
                                    best_cost = new_max
                                    best_r_temp = r
                                    best_p_temp = pos
                                elif new_max < second_best_cost:
                                    second_best_cost = new_max
                        regret = second_best_cost - best_cost
                        if regret > best_regret or (regret == best_regret and cust < best_cust):
                            best_regret = regret
                            best_cust = cust
                            best_r = best_r_temp
                            best_pos = best_p_temp
                            best_new_max = best_cost
                    current_routes[best_r].insert(best_pos, best_cust)
                    route_dists[best_r] = route_distance(current_routes[best_r])
                    unassigned.remove(best_cust)
                current_max = compute_max_dist(current_routes)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in current_routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
                improved = True
                no_improve_iter = 0
        if not improved:
            no_improve_iter += 1
            deviation *= 0.99
        else:
            no_improve_iter = 0
            deviation = 0.1 * current_max
        if no_improve_iter >= 20:
            # restart from scratch
            new_routes, new_route_dists = construct()
            new_max = compute_max_dist(new_routes)
            if new_max < best_max:
                best_max = new_max
                best_routes = [r[:] for r in new_routes]
                try:
                    report_best_vrp(best_routes)
                except NameError:
                    pass
            current_routes = new_routes
            route_dists = new_route_dists
            current_max = new_max
            no_improve_iter = 0
            deviation = 0.1 * current_max
            improved = True
        if not improved:
            pass
    
    result = []
    for r in best_routes:
        if len(r) >= 2 and r[0] == 0 and r[-1] == 0:
            result.append(r)
        else:
            result.append([0] + [c for c in r if c != 0] + [0])
    while len(result) < truck_count:
        result.append([0, 0])
    return result