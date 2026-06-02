import numpy as np
import math
from itertools import combinations

def report_best_vrp(routes):
    pass

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Construction: greedy insertion minimizing max route distance
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0 for _ in range(truck_count)]
    for c in customers:
        best_route = -1
        best_pos = -1
        best_new_max = float('inf')
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                new_len = route_lengths[r_idx] + increase
                new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_route = r_idx
                    best_pos = pos
        route = routes[best_route]
        route.insert(best_pos, c)
        delta = distance_matrix[route[best_pos-1], c] + distance_matrix[c, route[best_pos+1]] - distance_matrix[route[best_pos-1], route[best_pos+1]]
        route_lengths[best_route] += delta
    report_best_vrp(routes)

    # Intra-route 2-opt
    for r_idx in range(truck_count):
        route = routes[r_idx]
        if len(route) <= 3:
            continue
        improved = True
        max_iter = len(route) * 2
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new_d = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new_d < old:
                        route[i:j+1] = reversed(route[i:j+1])
                        route_lengths[r_idx] -= old - new_d
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break
        # Recompute for accuracy
        route_lengths[r_idx] = sum(distance_matrix[route[k], route[k+1]] for k in range(len(route)-1))

    # Inter-route improvement: relocate and swap (accept equal or better)
    total_customers = n - 1
    max_iter_inter = total_customers * truck_count * 2
    for _ in range(max_iter_inter):
        improved = False
        # Relocate
        for r_from in range(truck_count):
            route_from = routes[r_from]
            if len(route_from) <= 2:
                continue
            for idx_c in range(1, len(route_from)-1):
                c = route_from[idx_c]
                prev = route_from[idx_c-1]
                nxt = route_from[idx_c+1]
                cost_remove = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                new_len_from = route_lengths[r_from] - cost_remove
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    route_to = routes[r_to]
                    best_new_len_to = float('inf')
                    best_pos = -1
                    for pos in range(1, len(route_to)):
                        prev_to = route_to[pos-1]
                        nxt_to = route_to[pos]
                        cost_insert = distance_matrix[prev_to, c] + distance_matrix[c, nxt_to] - distance_matrix[prev_to, nxt_to]
                        new_len_to = route_lengths[r_to] + cost_insert
                        if new_len_to < best_new_len_to:
                            best_new_len_to = new_len_to
                            best_pos = pos
                    new_max = max(route_lengths[:r_from] + [new_len_from] + route_lengths[r_from+1:r_to] + [best_new_len_to] + route_lengths[r_to+1:])
                    current_max = max(route_lengths)
                    if new_max <= current_max:  # accept equal or better
                        route_from.pop(idx_c)
                        route_lengths[r_from] = new_len_from
                        route_to.insert(best_pos, c)
                        route_lengths[r_to] = best_new_len_to
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Swap
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                c1 = route1[idx1]
                prev1 = route1[idx1-1]
                nxt1 = route1[idx1+1]
                cost_remove1 = distance_matrix[prev1, c1] + distance_matrix[c1, nxt1] - distance_matrix[prev1, nxt1]
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 2:
                        continue
                    for idx2 in range(1, len(route2)-1):
                        c2 = route2[idx2]
                        prev2 = route2[idx2-1]
                        nxt2 = route2[idx2+1]
                        cost_remove2 = distance_matrix[prev2, c2] + distance_matrix[c2, nxt2] - distance_matrix[prev2, nxt2]
                        cost_insert1 = distance_matrix[prev1, c2] + distance_matrix[c2, nxt1] - distance_matrix[prev1, nxt1]
                        new_len1 = route_lengths[r1] - cost_remove1 + cost_insert1
                        cost_insert2 = distance_matrix[prev2, c1] + distance_matrix[c1, nxt2] - distance_matrix[prev2, nxt2]
                        new_len2 = route_lengths[r2] - cost_remove2 + cost_insert2
                        new_max = max(route_lengths[:r1] + [new_len1] + route_lengths[r1+1:r2] + [new_len2] + route_lengths[r2+1:])
                        current_max = max(route_lengths)
                        if new_max <= current_max:
                            del route1[idx1]
                            del route2[idx2]
                            route1.insert(idx1, c2)
                            route2.insert(idx2, c1)
                            route_lengths[r1] = new_len1
                            route_lengths[r2] = new_len2
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    # Ruin-and-recreate phase
    n_customers = n - 1
    max_ruin_iter = min(5, n_customers)
    best_routes = [list(route) for route in routes]
    best_max = max(route_lengths)
    for _ in range(max_ruin_iter):
        # Find route with max distance
        max_len_idx = max(range(truck_count), key=lambda i: route_lengths[i])
        route = routes[max_len_idx]
        if len(route) <= 3:  # only depot
            continue
        # Remove about half of customers from that route (excluding depot)
        remove_count = max(1, (len(route) - 2) // 2)
        # Remove customers that are farthest from depot (deterministic tie-breaking by index)
        indices = list(range(1, len(route)-1))
        # Sort by distance from depot descending, then index descending
        indices.sort(key=lambda i: (distance_matrix[0, route[i]], -route[i]), reverse=True)
        removed_indices = indices[:remove_count]
        removed_customers = [route[i] for i in removed_indices]
        # Remove them in reverse order to preserve indices
        for i in sorted(removed_indices, reverse=True):
            route.pop(i)
        # Recompute route length for affected route
        route_lengths[max_len_idx] = sum(distance_matrix[route[k], route[k+1]] for k in range(len(route)-1))
        # Reinsert removed customers in deterministic order (descending distance from depot, then index)
        removed_customers.sort(key=lambda c: (distance_matrix[0, c], c), reverse=True)
        for c in removed_customers:
            best_route = -1
            best_pos = -1
            best_new_max = float('inf')
            for r_idx, route_to in enumerate(routes):
                for pos in range(1, len(route_to)):
                    prev = route_to[pos-1]
                    nxt = route_to[pos]
                    increase = distance_matrix[prev, c] + distance_matrix[c, nxt] - distance_matrix[prev, nxt]
                    new_len = route_lengths[r_idx] + increase
                    new_max = max(route_lengths[:r_idx] + [new_len] + route_lengths[r_idx+1:])
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_route = r_idx
                        best_pos = pos
            route_to = routes[best_route]
            route_to.insert(best_pos, c)
            delta = distance_matrix[route_to[best_pos-1], c] + distance_matrix[c, route_to[best_pos+1]] - distance_matrix[route_to[best_pos-1], route_to[best_pos+1]]
            route_lengths[best_route] += delta
        report_best_vrp(routes)
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(route) for route in routes]
    # Restore best found
    routes = best_routes
    report_best_vrp(routes)
    return routes