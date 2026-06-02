import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count
    unassigned = list(range(1, n))
    
    # Construction: regret insertion minimizing max route distance
    while unassigned:
        best_cust = None
        best_regret = -1.0
        best_route = None
        best_pos = None
        best_max = None
        for cust in unassigned:
            best_max_val = float('inf')
            second_best_max = float('inf')
            candidate_route = None
            candidate_pos = None
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    increase = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_len = route_lengths[r_idx] + increase
                    new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                    if new_max < best_max_val:
                        second_best_max = best_max_val
                        best_max_val = new_max
                        candidate_route = r_idx
                        candidate_pos = pos
                    elif new_max < second_best_max:
                        second_best_max = new_max
            regret = second_best_max - best_max_val
            if regret > best_regret or (regret == best_regret and (best_max is None or best_max_val < best_max)):
                best_regret = regret
                best_cust = cust
                best_route = candidate_route
                best_pos = candidate_pos
                best_max = best_max_val
            elif regret == best_regret and best_max_val == best_max and cust < best_cust:
                best_cust = cust
                best_route = candidate_route
                best_pos = candidate_pos
        # Insert best_cust
        route = routes[best_route]
        prev = route[best_pos-1]
        nxt = route[best_pos]
        increase = distance_matrix[prev, best_cust] + distance_matrix[best_cust, nxt] - distance_matrix[prev, nxt]
        route.insert(best_pos, best_cust)
        route_lengths[best_route] += increase
        unassigned.remove(best_cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_lengths)
    report_best_vrp(best_routes)
    
    # Improvement
    max_iter = n * 5
    for _ in range(max_iter):
        improved = False
        # Relocate: move a customer from the longest route to another
        dists = route_lengths[:]
        longest_idx = max(range(truck_count), key=lambda i: dists[i])
        longest_route = routes[longest_idx]
        for i in range(1, len(longest_route)-1):
            cust = longest_route[i]
            prev = longest_route[i-1]
            nxt = longest_route[i+1]
            removal_delta = distance_matrix[prev, nxt] - distance_matrix[prev, cust] - distance_matrix[cust, nxt]
            new_len_long = dists[longest_idx] + removal_delta
            for j in range(truck_count):
                if j == longest_idx:
                    continue
                route_j = routes[j]
                for pos in range(1, len(route_j)):
                    prev_j = route_j[pos-1]
                    nxt_j = route_j[pos]
                    insert_delta = distance_matrix[prev_j, cust] + distance_matrix[cust, nxt_j] - distance_matrix[prev_j, nxt_j]
                    new_len_j = dists[j] + insert_delta
                    # new max for all routes
                    new_max = max(new_len_long, new_len_j, max(dists[k] for k in range(truck_count) if k not in (longest_idx, j)))
                    if new_max < best_max:
                        # Apply move
                        longest_route.pop(i)
                        route_j.insert(pos, cust)
                        route_lengths[longest_idx] = new_len_long
                        route_lengths[j] = new_len_j
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Swap: exchange customers between two routes
        for i in range(truck_count):
            route_i = routes[i]
            if len(route_i) <= 2:
                continue
            for pos_i in range(1, len(route_i)-1):
                cust_i = route_i[pos_i]
                prev_i = route_i[pos_i-1]
                nxt_i = route_i[pos_i+1]
                delta_i_rem = distance_matrix[prev_i, nxt_i] - distance_matrix[prev_i, cust_i] - distance_matrix[cust_i, nxt_i]
                for j in range(i+1, truck_count):
                    route_j = routes[j]
                    if len(route_j) <= 2:
                        continue
                    for pos_j in range(1, len(route_j)-1):
                        cust_j = route_j[pos_j]
                        prev_j = route_j[pos_j-1]
                        nxt_j = route_j[pos_j+1]
                        delta_j_rem = distance_matrix[prev_j, nxt_j] - distance_matrix[prev_j, cust_j] - distance_matrix[cust_j, nxt_j]
                        # insert cust_j into route_i at pos_i
                        add_i = distance_matrix[prev_i, cust_j] + distance_matrix[cust_j, nxt_i] - distance_matrix[prev_i, nxt_i]
                        new_len_i = route_lengths[i] + delta_i_rem + add_i
                        # insert cust_i into route_j at pos_j
                        add_j = distance_matrix[prev_j, cust_i] + distance_matrix[cust_i, nxt_j] - distance_matrix[prev_j, nxt_j]
                        new_len_j = route_lengths[j] + delta_j_rem + add_j
                        new_max = max(new_len_i, new_len_j, max(route_lengths[k] for k in range(truck_count) if k not in (i, j)))
                        if new_max < best_max:
                            # Apply swap: first remove both (order matters for indices)
                            route_i.pop(pos_i)
                            route_j.pop(pos_j)
                            # Insert swapped customers
                            route_i.insert(pos_i, cust_j)
                            route_j.insert(pos_j, cust_i)
                            route_lengths[i] = new_len_i
                            route_lengths[j] = new_len_j
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 2-opt within routes
        for i in range(truck_count):
            route = routes[i]
            if len(route) <= 3:
                continue
            for a in range(0, len(route)-2):
                for b in range(a+1, len(route)-1):
                    delta = distance_matrix[route[a], route[b]] + distance_matrix[route[a+1], route[b+1]] - distance_matrix[route[a], route[a+1]] - distance_matrix[route[b], route[b+1]]
                    new_len = route_lengths[i] + delta
                    new_max = max(new_len, max(route_lengths[k] for k in range(truck_count) if k != i))
                    if new_max < best_max:
                        route[a+1:b+1] = reversed(route[a+1:b+1])
                        route_lengths[i] = new_len
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best_routes