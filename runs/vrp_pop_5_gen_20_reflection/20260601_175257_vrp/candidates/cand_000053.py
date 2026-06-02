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
    max_restarts = min(10, n * truck_count)
    no_improve_restarts = 0

    for restart in range(max_restarts):
        random.seed(restart)
        # Regret-2 construction
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0] * truck_count
        unassigned = set(customers)
        while unassigned:
            best_cust = None
            best_route = None
            best_pos = None
            best_cost = float('inf')
            best_regret = -float('inf')
            for cust in list(unassigned):
                costs = []
                for ri, route in enumerate(routes):
                    cur_len = route_lengths[ri]
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        add = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                        new_len = cur_len + add
                        new_max = max(route_lengths[:ri] + [new_len] + route_lengths[ri+1:])
                        costs.append((new_max, ri, pos, new_len))
                if not costs:
                    continue
                costs.sort(key=lambda x: x[0])
                best = costs[0][0]
                second_best = costs[1][0] if len(costs) > 1 else best
                regret = second_best - best
                if regret > best_regret or (regret == best_regret and best < best_cost):
                    best_regret = regret
                    best_cost = best
                    best_cust = cust
                    best_route = costs[0][1]
                    best_pos = costs[0][2]
            if best_cust is None:
                break
            route = routes[best_route]
            prev = route[best_pos-1]
            nxt = route[best_pos]
            route.insert(best_pos, best_cust)
            route_lengths[best_route] += distance_matrix[prev][best_cust] + distance_matrix[best_cust][nxt] - distance_matrix[prev][nxt]
            unassigned.remove(best_cust)

        best_routes = [list(r) for r in routes]
        best_max = max(route_lengths)
        report_best_vrp(best_routes)

        # Local search helper
        def local_search(routes, route_lengths, best_max):
            improved = True
            max_passes = min(100, n * truck_count)
            for _ in range(max_passes):
                if not improved:
                    break
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
                                    report_best_vrp(routes)
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
                                    report_best_vrp(routes)
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
                            delta = distance_matrix[route[a]][route[b]] + distance_matrix[route[a+1]][route[b+1]] - distance_matrix[route[a]][route[a+1]] - distance_matrix[route[b]][route[b+1]]
                            new_len = route_lengths[i] + delta
                            if new_len < best_max:
                                new_max = max(new_len, max(route_lengths[k] for k in range(truck_count) if k != i))
                                if new_max < best_max:
                                    route[a+1:b+1] = reversed(route[a+1:b+1])
                                    route_lengths[i] = new_len
                                    best_max = new_max
                                    report_best_vrp(routes)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            return routes, route_lengths, best_max, improved

        routes, route_lengths, best_max, _ = local_search(routes, route_lengths, best_max)
        if best_max < best_overall_max:
            best_overall = [list(r) for r in routes]
            best_overall_max = best_max
            no_improve_restarts = 0
        else:
            no_improve_restarts += 1

        # Adaptive perturbation if stuck
        if no_improve_restarts >= 2 and best_overall is not None:
            perturb_routes = [list(r) for r in best_overall]
            perturb_lengths = []
            for r in perturb_routes:
                l = 0.0
                for idx in range(len(r)-1):
                    l += distance_matrix[r[idx]][r[idx+1]]
                perturb_lengths.append(l)
            all_custs = []
            for r in perturb_routes:
                for c in r[1:-1]:
                    all_custs.append(c)
            fraction = random.uniform(0.2, 0.4)
            to_remove = int(len(all_custs) * fraction)
            if to_remove > 0:
                random.shuffle(all_custs)
                removed = all_custs[:to_remove]
                # Rebuild routes without removed customers
                new_routes = []
                new_lengths = []
                for r in perturb_routes:
                    new_route = [0]
                    for c in r[1:-1]:
                        if c not in removed:
                            new_route.append(c)
                    new_route.append(0)
                    new_routes.append(new_route)
                    l = 0.0
                    for idx in range(len(new_route)-1):
                        l += distance_matrix[new_route[idx]][new_route[idx+1]]
                    new_lengths.append(l)
                # Reinsert removed customers using regret
                unassigned = set(removed)
                while unassigned:
                    best_cust = None
                    best_route = None
                    best_pos = None
                    best_cost = float('inf')
                    best_regret = -float('inf')
                    for cust in list(unassigned):
                        costs = []
                        for ri, route in enumerate(new_routes):
                            cur_len = new_lengths[ri]
                            for pos in range(1, len(route)):
                                prev = route[pos-1]
                                nxt = route[pos]
                                add = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                                new_len = cur_len + add
                                new_max = max(new_lengths[:ri] + [new_len] + new_lengths[ri+1:])
                                costs.append((new_max, ri, pos, new_len))
                        if not costs:
                            continue
                        costs.sort(key=lambda x: x[0])
                        best = costs[0][0]
                        second_best = costs[1][0] if len(costs) > 1 else best
                        regret = second_best - best
                        if regret > best_regret or (regret == best_regret and best < best_cost):
                            best_regret = regret
                            best_cost = best
                            best_cust = cust
                            best_route = costs[0][1]
                            best_pos = costs[0][2]
                    if best_cust is None:
                        break
                    route = new_routes[best_route]
                    prev = route[best_pos-1]
                    nxt = route[best_pos]
                    route.insert(best_pos, best_cust)
                    new_lengths[best_route] += distance_matrix[prev][best_cust] + distance_matrix[best_cust][nxt] - distance_matrix[prev][nxt]
                    unassigned.remove(best_cust)
                routes = new_routes
                route_lengths = new_lengths
                best_max = max(route_lengths)
                if best_max < best_overall_max:
                    best_overall = [list(r) for r in routes]
                    best_overall_max = best_max
                    report_best_vrp(best_overall)
                routes, route_lengths, best_max, _ = local_search(routes, route_lengths, best_max)
                if best_max < best_overall_max:
                    best_overall = [list(r) for r in routes]
                    best_overall_max = best_max
                    report_best_vrp(best_overall)
                    no_improve_restarts = 0
                else:
                    no_improve_restarts = 0

    if best_overall is None:
        best_overall = [[0, 0] for _ in range(truck_count)]
    return best_overall