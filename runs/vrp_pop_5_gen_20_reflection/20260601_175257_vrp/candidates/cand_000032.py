import numpy as np
import math
import random
import heapq
import itertools
import collections
import time

def report_best_vrp(routes):
    pass

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    customers = list(range(1, n))
    best_overall = None
    best_overall_max = float('inf')

    def regret_insertion(seed):
        random.seed(seed)
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0] * truck_count
        unassigned = customers[:]
        random.shuffle(unassigned)
        while unassigned:
            best_cust = None
            best_max = float('inf')
            best_route = None
            best_pos = None
            best_len = None
            best_regret = -1
            for cust in unassigned:
                insert_costs = []
                for ri, route in enumerate(routes):
                    cur_len = route_lengths[ri]
                    min_cost = float('inf')
                    min_pos = None
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        add = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                        new_len = cur_len + add
                        if new_len < min_cost:
                            min_cost = new_len
                            min_pos = pos
                    insert_costs.append((min_cost, min_pos, ri))
                # regret-2: difference between best and second best insertion cost
                sorted_costs = sorted(insert_costs, key=lambda x: x[0])
                regret = sorted_costs[1][0] - sorted_costs[0][0] if len(sorted_costs) > 1 else sorted_costs[0][0]
                # also compute resulting max if inserted in best route
                best_route_candidate = sorted_costs[0][2]
                best_cost, best_pos_candidate, _ = sorted_costs[0]
                new_len_candidate = route_lengths[best_route_candidate] + best_cost
                max_candidate = max(route_lengths[:best_route_candidate] + [new_len_candidate] + route_lengths[best_route_candidate+1:])
                # tie-break by lower max then lower route length
                if max_candidate < best_max or (max_candidate == best_max and (best_len is None or new_len_candidate < best_len)):
                    best_max = max_candidate
                    best_cust = cust
                    best_route = best_route_candidate
                    best_pos = best_pos_candidate
                    best_len = new_len_candidate
                    best_regret = regret
            # assign best_cust
            route = routes[best_route]
            route.insert(best_pos, best_cust)
            route_lengths[best_route] = best_len
            unassigned.remove(best_cust)
        return routes, route_lengths

    def local_search(routes, route_lengths, current_max):
        max_passes = min(100, n * truck_count)
        improved_global = False
        for _ in range(max_passes):
            improved = False
            # Inter-route relocate
            for i in range(truck_count):
                if len(routes[i]) <= 2:
                    continue
                for pos_i in range(1, len(routes[i])-1):
                    cust = routes[i][pos_i]
                    prev_i = routes[i][pos_i-1]
                    next_i = routes[i][pos_i+1]
                    removal = distance_matrix[prev_i][next_i] - distance_matrix[prev_i][cust] - distance_matrix[cust][next_i]
                    new_len_i = route_lengths[i] + removal
                    for j in range(truck_count):
                        if j == i:
                            continue
                        route_j = routes[j]
                        for pos_j in range(1, len(route_j)):
                            prev_j = route_j[pos_j-1]
                            next_j = route_j[pos_j]
                            insert = distance_matrix[prev_j][cust] + distance_matrix[cust][next_j] - distance_matrix[prev_j][next_j]
                            new_len_j = route_lengths[j] + insert
                            new_max = max(new_len_i, new_len_j, max(route_lengths[k] for k in range(truck_count) if k not in (i, j)))
                            if new_max < current_max:
                                routes[i].pop(pos_i)
                                route_j.insert(pos_j, cust)
                                route_lengths[i] = new_len_i
                                route_lengths[j] = new_len_j
                                current_max = new_max
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                improved_global = True
                continue
            # Inter-route swap
            for i in range(truck_count):
                if len(routes[i]) <= 2:
                    continue
                for pos_i in range(1, len(routes[i])-1):
                    cust_i = routes[i][pos_i]
                    prev_i = routes[i][pos_i-1]
                    next_i = routes[i][pos_i+1]
                    delta_i = distance_matrix[prev_i][next_i] - distance_matrix[prev_i][cust_i] - distance_matrix[cust_i][next_i]
                    for j in range(i+1, truck_count):
                        if len(routes[j]) <= 2:
                            continue
                        for pos_j in range(1, len(routes[j])-1):
                            cust_j = routes[j][pos_j]
                            prev_j = routes[j][pos_j-1]
                            next_j = routes[j][pos_j+1]
                            delta_j = distance_matrix[prev_j][next_j] - distance_matrix[prev_j][cust_j] - distance_matrix[cust_j][next_j]
                            add_i = distance_matrix[prev_i][cust_j] + distance_matrix[cust_j][next_i] - distance_matrix[prev_i][next_i]
                            add_j = distance_matrix[prev_j][cust_i] + distance_matrix[cust_i][next_j] - distance_matrix[prev_j][next_j]
                            new_len_i = route_lengths[i] + delta_i + add_i
                            new_len_j = route_lengths[j] + delta_j + add_j
                            new_max = max(new_len_i, new_len_j, max(route_lengths[k] for k in range(truck_count) if k not in (i, j)))
                            if new_max < current_max:
                                routes[i].pop(pos_i)
                                routes[j].pop(pos_j)
                                routes[i].insert(pos_i, cust_j)
                                routes[j].insert(pos_j, cust_i)
                                route_lengths[i] = new_len_i
                                route_lengths[j] = new_len_j
                                current_max = new_max
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                improved_global = True
                continue
            # 2-opt within routes
            for i in range(truck_count):
                route = routes[i]
                if len(route) <= 3:
                    continue
                for a in range(0, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        delta = distance_matrix[route[a]][route[b]] + distance_matrix[route[a+1]][route[b+1]] - distance_matrix[route[a]][route[a+1]] - distance_matrix[route[b]][route[b+1]]
                        new_len = route_lengths[i] + delta
                        if new_len < current_max:
                            new_max = max(new_len, max(route_lengths[k] for k in range(truck_count) if k != i))
                            if new_max < current_max:
                                route[a+1:b+1] = reversed(route[a+1:b+1])
                                route_lengths[i] = new_len
                                current_max = new_max
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return current_max, improved_global

    def perturb(routes, route_lengths, strength=0.3):
        # move customers from longest routes to shortest
        n_cust = sum(len(r)-2 for r in routes)
        if n_cust == 0:
            return
        num_move = max(1, int(n_cust * strength))
        # identify route indices sorted by length
        sorted_indices = sorted(range(truck_count), key=lambda i: route_lengths[i], reverse=True)
        # pick customers from longest routes
        to_move = []
        for idx in sorted_indices:
            r = routes[idx]
            if len(r) <= 2:
                continue
            # select random customer from route
            pos = random.randint(1, len(r)-2)
            cust = r[pos]
            to_move.append((cust, idx, pos))
            if len(to_move) >= num_move:
                break
        # remove them
        for cust, idx, pos in to_move:
            r = routes[idx]
            prev = r[pos-1]
            nxt = r[pos+1]
            removal = distance_matrix[prev][nxt] - distance_matrix[prev][cust] - distance_matrix[cust][nxt]
            route_lengths[idx] += removal
            r.pop(pos)
        # re-insert in cheapest position among shortest routes
        random.shuffle(to_move)
        for cust, _, _ in to_move:
            best_max = float('inf')
            best_route = None
            best_pos = None
            best_len = None
            for ri, route in enumerate(routes):
                cur_len = route_lengths[ri]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    add = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                    new_len = cur_len + add
                    new_max = max(route_lengths[:ri] + [new_len] + route_lengths[ri+1:])
                    if new_max < best_max or (new_max == best_max and (best_len is None or new_len < best_len)):
                        best_max = new_max
                        best_route = ri
                        best_pos = pos
                        best_len = new_len
            route = routes[best_route]
            prev = route[best_pos-1]
            nxt = route[best_pos]
            route.insert(best_pos, cust)
            route_lengths[best_route] += distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]

    max_restarts = min(5, n * truck_count)
    for restart in range(max_restarts):
        routes, route_lengths = regret_insertion(restart)
        current_max = max(route_lengths)
        best_routes_local = [list(r) for r in routes]
        best_max_local = current_max
        report_best_vrp(best_routes_local)
        new_max, improved = local_search(routes, route_lengths, current_max)
        if improved:
            best_max_local = new_max
            best_routes_local = [list(r) for r in routes]
            report_best_vrp(best_routes_local)
        for _ in range(3):
            perturb(routes, route_lengths, 0.3)
            new_max, improved = local_search(routes, route_lengths, best_max_local)
            if improved:
                best_max_local = new_max
                best_routes_local = [list(r) for r in routes]
                report_best_vrp(best_routes_local)
        if best_max_local < best_overall_max:
            best_overall = [list(r) for r in best_routes_local]
            best_overall_max = best_max_local

    if best_overall is None:
        best_overall = [[0, 0] for _ in range(truck_count)]
    return best_overall