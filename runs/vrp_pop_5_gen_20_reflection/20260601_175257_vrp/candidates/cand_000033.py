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
        unassigned = customers[:]
        random.shuffle(unassigned)
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0] * truck_count
        # initial assignment: each route gets one customer to start?
        # Actually, let's assign first truck_count customers to each route greedily
        if truck_count <= len(unassigned):
            for i in range(truck_count):
                cust = unassigned.pop(0)
                routes[i].insert(1, cust)
                route_lengths[i] = distance_matrix[0][cust] + distance_matrix[cust][0]
        else:
            # fewer customers than trucks, assign all to first routes
            for i in range(len(unassigned)):
                cust = unassigned.pop(0)
                routes[i].insert(1, cust)
                route_lengths[i] = distance_matrix[0][cust] + distance_matrix[cust][0]
            while len(unassigned) > 0:
                cust = unassigned.pop(0)
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
                route_lengths[best_route] = best_len
            return routes, route_lengths

        # Regret-2 insertion for remaining customers
        while unassigned:
            # For each unassigned customer, compute best and second best insertion cost (added distance) into any route position
            best_costs = []
            second_best_costs = []
            best_route_info = []  # (route, pos, new_len, new_max)
            for cust in unassigned:
                costs = []
                infos = []
                for ri, route in enumerate(routes):
                    cur_len = route_lengths[ri]
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        add = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                        new_len = cur_len + add
                        new_max = max(route_lengths[:ri] + [new_len] + route_lengths[ri+1:])
                        costs.append(new_max)
                        infos.append((ri, pos, new_len, new_max))
                if len(costs) >= 2:
                    # find two smallest costs
                    sorted_indices = sorted(range(len(costs)), key=lambda i: costs[i])
                    best_idx = sorted_indices[0]
                    second_idx = sorted_indices[1]
                    best_cost = costs[best_idx]
                    second_cost = costs[second_idx]
                    best_info = infos[best_idx]
                elif len(costs) == 1:
                    best_cost = costs[0]
                    second_cost = float('inf')
                    best_info = infos[0]
                else:
                    continue
                regret = second_cost - best_cost
                best_costs.append(best_cost)
                second_best_costs.append(second_cost)
                best_route_info.append(best_info)
            # Select customer with maximum regret
            max_regret = -float('inf')
            best_cust_idx = -1
            for idx, cust in enumerate(unassigned):
                if best_costs[idx] == float('inf'):
                    continue
                regret = second_best_costs[idx] - best_costs[idx] if second_best_costs[idx] != float('inf') else 0
                if regret > max_regret or (regret == max_regret and best_costs[idx] < best_costs[best_cust_idx]):
                    max_regret = regret
                    best_cust_idx = idx
            if best_cust_idx == -1:
                break
            cust = unassigned.pop(best_cust_idx)
            ri, pos, new_len, new_max = best_route_info[best_cust_idx]
            route = routes[ri]
            route.insert(pos, cust)
            route_lengths[ri] = new_len
        return routes, route_lengths

    def perturb(routes, route_lengths, strength):
        n_cust = sum(len(r)-2 for r in routes)
        if n_cust == 0:
            return
        num_move = max(1, int(n_cust * strength))
        all_custs = []
        cust_route = {}
        for i, r in enumerate(routes):
            for c in r[1:-1]:
                all_custs.append(c)
                cust_route[c] = i
        to_move = random.sample(all_custs, min(num_move, len(all_custs)))
        for c in to_move:
            ri = cust_route[c]
            r = routes[ri]
            pos = r.index(c)
            prev = r[pos-1]
            nxt = r[pos+1]
            removal_delta = distance_matrix[prev][nxt] - distance_matrix[prev][c] - distance_matrix[c][nxt]
            route_lengths[ri] += removal_delta
            r.pop(pos)
        # Reinsert using regret-2 insertion
        # Build list of unassigned
        unassigned = to_move[:]
        random.shuffle(unassigned)
        # Compute best insertion for each
        while unassigned:
            best_costs = []
            second_best_costs = []
            best_route_info = []
            for cust in unassigned:
                costs = []
                infos = []
                for ri, route in enumerate(routes):
                    cur_len = route_lengths[ri]
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        add = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                        new_len = cur_len + add
                        new_max = max(route_lengths[:ri] + [new_len] + route_lengths[ri+1:])
                        costs.append(new_max)
                        infos.append((ri, pos, new_len, new_max))
                if len(costs) >= 2:
                    sorted_indices = sorted(range(len(costs)), key=lambda i: costs[i])
                    best_idx = sorted_indices[0]
                    second_idx = sorted_indices[1]
                    best_cost = costs[best_idx]
                    second_cost = costs[second_idx]
                    best_info = infos[best_idx]
                elif len(costs) == 1:
                    best_cost = costs[0]
                    second_cost = float('inf')
                    best_info = infos[0]
                else:
                    continue
                regret = second_cost - best_cost
                best_costs.append(best_cost)
                second_best_costs.append(second_cost)
                best_route_info.append(best_info)
            max_regret = -float('inf')
            best_cust_idx = -1
            for idx, cust in enumerate(unassigned):
                if best_costs[idx] == float('inf'):
                    continue
                regret = second_best_costs[idx] - best_costs[idx] if second_best_costs[idx] != float('inf') else 0
                if regret > max_regret or (regret == max_regret and best_costs[idx] < best_costs[best_cust_idx]):
                    max_regret = regret
                    best_cust_idx = idx
            if best_cust_idx == -1:
                break
            cust = unassigned.pop(best_cust_idx)
            ri, pos, new_len, new_max = best_route_info[best_cust_idx]
            route = routes[ri]
            route.insert(pos, cust)
            route_lengths[ri] = new_len

    def local_search(routes, route_lengths, best_max):
        max_passes = min(100, n * truck_count)
        improved_global = False
        for _ in range(max_passes):
            improved = False
            # Inter-route relocate
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i)-1):
                    cust = route_i[pos_i]
                    prev_i = route_i[pos_i-1]
                    next_i = route_i[pos_i+1]
                    removal_delta = distance_matrix[prev_i][next_i] - distance_matrix[prev_i][cust] - distance_matrix[cust][next_i]
                    new_len_i = route_lengths[i] + removal_delta
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

            # Inter-route swap
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

    max_restarts = min(10, 2 * n)
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
        strength = 0.2
        for cycle in range(5):
            perturb(routes, route_lengths, strength)
            new_max, improved = local_search(routes, route_lengths, best_max_local)
            if improved:
                best_max_local = new_max
                best_routes_local = [list(r) for r in routes]
                report_best_vrp(best_routes_local)
            else:
                strength = min(0.5, strength + 0.05)
        if best_max_local < best_overall_max:
            best_overall = [list(r) for r in best_routes_local]
            best_overall_max = best_max_local

    if best_overall is None:
        best_overall = [[0, 0] for _ in range(truck_count)]
    return best_overall