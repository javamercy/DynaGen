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
        # Initialize routes with depot
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0] * truck_count
        unassigned = set(customers)
        # First assign two customers to each truck to get started
        for truck in range(truck_count):
            if not unassigned:
                break
            # Find farthest customer from depot among unassigned (or random)
            best_cust = max(unassigned, key=lambda c: distance_matrix[0][c])
            unassigned.remove(best_cust)
            routes[truck].insert(1, best_cust)
            route_lengths[truck] = 2 * distance_matrix[0][best_cust]
        # Now assign remaining customers via regret-2
        while unassigned:
            # For each unassigned customer, compute insertion cost into each route and regret
            best_cust = None
            best_regret = -float('inf')
            best_route = None
            best_pos = None
            best_max = None
            for cust in list(unassigned):
                costs = []
                for ri, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        add = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                        new_len = route_lengths[ri] + add
                        new_max = max(route_lengths[:ri] + [new_len] + route_lengths[ri+1:])
                        costs.append((new_max, add, ri, pos))
                # Sort by new_max ascending, then by add ascending
                costs.sort(key=lambda x: (x[0], x[1]))
                # Regret = cost_best - cost_second_best (by new_max)
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = 0
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_route = costs[0][2]
                    best_pos = costs[0][3]
                    best_max = costs[0][0]
                elif regret == best_regret and best_cust is not None:
                    # Tie-break: prefer lower best_max
                    if costs[0][0] < best_max:
                        best_cust = cust
                        best_route = costs[0][2]
                        best_pos = costs[0][3]
                        best_max = costs[0][0]
            # Insert best_cust
            unassigned.remove(best_cust)
            route = routes[best_route]
            prev = route[best_pos-1]
            nxt = route[best_pos]
            add = distance_matrix[prev][best_cust] + distance_matrix[best_cust][nxt] - distance_matrix[prev][nxt]
            route.insert(best_pos, best_cust)
            route_lengths[best_route] += add
        return routes, route_lengths

    def perturb(routes, route_lengths, strength=None):
        if strength is None:
            strength = random.uniform(0.1, 0.4)
        n_cust = sum(len(r)-2 for r in routes)
        num_move = max(1, int(n_cust * strength))
        # Collect all customers
        all_custs = []
        cust_route = {}
        for i, r in enumerate(routes):
            for c in r[1:-1]:
                all_custs.append(c)
                cust_route[c] = i
        # Remove customers (prefer from longer routes)
        to_move = []
        if num_move <= len(all_custs):
            # Weight removal probability by route length
            weights = []
            for c in all_custs:
                ri = cust_route[c]
                weights.append(route_lengths[ri] + 1e-9)  # avoid zero
            total = sum(weights)
            probs = [w/total for w in weights]
            to_move = random.choices(all_custs, weights=probs, k=num_move)
        else:
            to_move = all_custs[:]
        # Remove selected customers
        for c in to_move:
            ri = cust_route[c]
            r = routes[ri]
            pos = r.index(c)
            prev = r[pos-1]
            nxt = r[pos+1]
            removal_delta = distance_matrix[prev][nxt] - distance_matrix[prev][c] - distance_matrix[c][nxt]
            route_lengths[ri] += removal_delta
            r.pop(pos)
        # Reinsert randomly shuffled
        random.shuffle(to_move)
        for cust in to_move:
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

    def local_search(routes, route_lengths, best_max):
        max_passes = min(100, n * truck_count)
        improved_global = False
        for _ in range(max_passes):
            improved = False
            # Inter-route relocate (highest max route first)
            order = list(range(truck_count))
            order.sort(key=lambda i: route_lengths[i], reverse=True)
            for i in order:
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust = route_i[pos_i]
                    prev_i = route_i[pos_i-1]
                    next_i = route_i[pos_i+1]
                    removal_delta = distance_matrix[prev_i][next_i] - distance_matrix[prev_i][cust] - distance_matrix[cust][next_i]
                    new_len_i = route_lengths[i] + removal_delta
                    # Try to insert into routes with smaller max to reduce overall max
                    for j in range(truck_count):
                        if j == i:
                            continue
                        route_j = routes[j]
                        for pos_j in range(1, len(route_j)):
                            prev_j = route_j[pos_j-1]
                            next_j = route_j[pos_j]
                            insert_delta = distance_matrix[prev_j][cust] + distance_matrix[cust][next_j] - distance_matrix[prev_j][next_j]
                            new_len_j = route_lengths[j] + insert_delta
                            new_max = max(new_len_i, new_len_j, max(route_lengths[k] for k in range(truck_count) if k not in (i, j)))
                            if new_max < best_max:
                                route_i.pop(pos_i)
                                route_j.insert(pos_j, cust)
                                route_lengths[i] = new_len_i
                                route_lengths[j] = new_len_j
                                best_max = new_max
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

            # Inter-route swap (consider pairs of routes)
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust_i = route_i[pos_i]
                    prev_i = route_i[pos_i-1]
                    next_i = route_i[pos_i+1]
                    delta_i_rem = distance_matrix[prev_i][next_i] - distance_matrix[prev_i][cust_i] - distance_matrix[cust_i][next_i]
                    for j in range(i+1, truck_count):
                        route_j = routes[j]
                        if len(route_j) <= 2:
                            continue
                        for pos_j in range(1, len(route_j)-1):
                            cust_j = route_j[pos_j]
                            prev_j = route_j[pos_j-1]
                            next_j = route_j[pos_j+1]
                            delta_j_rem = distance_matrix[prev_j][next_j] - distance_matrix[prev_j][cust_j] - distance_matrix[cust_j][next_j]
                            add_i = distance_matrix[prev_i][cust_j] + distance_matrix[cust_j][next_i] - distance_matrix[prev_i][next_i]
                            add_j = distance_matrix[prev_j][cust_i] + distance_matrix[cust_i][next_j] - distance_matrix[prev_j][next_j]
                            new_len_i = route_lengths[i] + delta_i_rem + add_i
                            new_len_j = route_lengths[j] + delta_j_rem + add_j
                            new_max = max(new_len_i, new_len_j, max(route_lengths[k] for k in range(truck_count) if k not in (i, j)))
                            if new_max < best_max:
                                route_i.pop(pos_i)
                                route_j.pop(pos_j)
                                route_i.insert(pos_i, cust_j)
                                route_j.insert(pos_j, cust_i)
                                route_lengths[i] = new_len_i
                                route_lengths[j] = new_len_j
                                best_max = new_max
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
                        if new_len < best_max:
                            new_max = max(new_len, max(route_lengths[k] for k in range(truck_count) if k != i))
                            if new_max < best_max:
                                route[a+1:b+1] = reversed(route[a+1:b+1])
                                route_lengths[i] = new_len
                                best_max = new_max
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return best_max, improved_global

    max_restarts = min(5, n * truck_count)
    for restart in range(max_restarts):
        routes, route_lengths = regret_insertion(restart)
        best_max_local = max(route_lengths)
        best_routes_local = [list(r) for r in routes]
        report_best_vrp(best_routes_local)
        new_max, improved = local_search(routes, route_lengths, best_max_local)
        if improved:
            best_max_local = new_max
            best_routes_local = [list(r) for r in routes]
            report_best_vrp(best_routes_local)
        # Perturb-reoptimize cycles with varying strength
        for cycle in range(4):
            strength = random.uniform(0.1, 0.4)
            perturb(routes, route_lengths, strength)
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