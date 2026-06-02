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
    for restart in range(max_restarts):
        random.seed(restart)
        if restart == 0 or best_overall is None:
            # random permutation insertion
            shuffled = list(customers)
            random.shuffle(shuffled)
            routes = [[0, 0] for _ in range(truck_count)]
            route_lengths = [0.0] * truck_count
            for cust in shuffled:
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
        else:
            # perturb best solution
            routes = [list(r) for r in best_overall]
            route_lengths = [0.0] * truck_count
            for ri, route in enumerate(routes):
                for k in range(len(route)-1):
                    route_lengths[ri] += distance_matrix[route[k]][route[k+1]]
            # choose a random customer from a random route (not depot)
            nonempty = [ri for ri, r in enumerate(routes) if len(r) > 2]
            if nonempty:
                ri = random.choice(nonempty)
                route_i = routes[ri]
                pos = random.randint(1, len(route_i)-2)
                cust = route_i.pop(pos)
                # remove contribution
                prev = route_i[pos-1]
                nxt = route_i[pos] if pos < len(route_i) else 0
                route_lengths[ri] -= distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                # insert into random position in any route
                rj = random.randint(0, truck_count-1)
                route_j = routes[rj]
                pos_j = random.randint(1, len(route_j)-1)
                # ideally insert in position that doesn't break feasibility
                # simple: insert at random
                prev_j = route_j[pos_j-1]
                nxt_j = route_j[pos_j]
                route_j.insert(pos_j, cust)
                route_lengths[rj] += distance_matrix[prev_j][cust] + distance_matrix[cust][nxt_j] - distance_matrix[prev_j][nxt_j]
            else:
                # all empty? shouldn't happen
                pass

        # local search
        best_routes = [list(r) for r in routes]
        best_max = max(route_lengths)
        report_best_vrp(best_routes)
        max_passes = min(200, n * truck_count)
        for _ in range(max_passes):
            improved = False
            # inter-route relocate
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
            # inter-route swap
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
                        delta = distance_matrix[route[a]][route[b]] + distance_matrix[route[a+1]][route[b+1]] - distance_matrix[route[a]][route[a+1]] - distance_matrix[route[b]][route[b+1]]
                        new_len = route_lengths[i] + delta
                        if new_len < best_max:
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
        # after local search, update best overall
        if best_max < best_overall_max:
            best_overall = [list(r) for r in best_routes]
            best_overall_max = best_max
    if best_overall is None:
        best_overall = [[0, 0] for _ in range(truck_count)]
    return best_overall