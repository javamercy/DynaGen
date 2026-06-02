import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]

    best_routes = None
    best_max = float('inf')

    num_restarts = min(3, n - 1)
    for restart in range(num_restarts):
        # Initialize empty routes
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0] * truck_count
        unassigned = set(customers)

        # Cheapest insertion heuristic: repeatedly insert the customer that minimizes the max route distance increase
        while unassigned:
            best_cust = None
            best_route_idx = None
            best_pos = None
            best_max_temp = float('inf')
            for cust in unassigned:
                for ri in range(truck_count):
                    route = routes[ri]
                    cur_len = route_lengths[ri]
                    for pos in range(1, len(route)):
                        prev = route[pos - 1]
                        nxt = route[pos]
                        add = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                        new_len = cur_len + add
                        new_max = max(route_lengths[:ri] + [new_len] + route_lengths[ri+1:])
                        if new_max < best_max_temp:
                            best_max_temp = new_max
                            best_cust = cust
                            best_route_idx = ri
                            best_pos = pos
            # Insert the best customer
            route = routes[best_route_idx]
            prev = route[best_pos - 1]
            nxt = route[best_pos]
            route.insert(best_pos, best_cust)
            route_lengths[best_route_idx] += distance_matrix[prev][best_cust] + distance_matrix[best_cust][nxt] - distance_matrix[prev][nxt]
            unassigned.remove(best_cust)

        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

        # Local search: first-improvement on max route distance
        max_passes = min(100, n * truck_count)
        for _ in range(max_passes):
            improved = False
            # Inter-route relocate
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i) - 1):
                    cust = route_i[pos_i]
                    prev_i = route_i[pos_i - 1]
                    next_i = route_i[pos_i + 1]
                    removal_delta = distance_matrix[prev_i][next_i] - distance_matrix[prev_i][cust] - distance_matrix[cust][next_i]
                    new_len_i = route_lengths[i] + removal_delta
                    for j in range(truck_count):
                        if j == i:
                            continue
                        route_j = routes[j]
                        for pos_j in range(1, len(route_j)):
                            prev_j = route_j[pos_j - 1]
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
            # Inter-route swap
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for pos_i in range(1, len(route_i) - 1):
                    cust_i = route_i[pos_i]
                    prev_i = route_i[pos_i - 1]
                    next_i = route_i[pos_i + 1]
                    delta_i_rem = distance_matrix[prev_i][next_i] - distance_matrix[prev_i][cust_i] - distance_matrix[cust_i][next_i]
                    for j in range(i + 1, truck_count):
                        route_j = routes[j]
                        if len(route_j) <= 2:
                            continue
                        for pos_j in range(1, len(route_j) - 1):
                            cust_j = route_j[pos_j]
                            prev_j = route_j[pos_j - 1]
                            next_j = route_j[pos_j + 1]
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
                for a in range(0, len(route) - 2):
                    for b in range(a + 1, len(route) - 1):
                        delta = distance_matrix[route[a]][route[b]] + distance_matrix[route[a + 1]][route[b + 1]] - distance_matrix[route[a]][route[a + 1]] - distance_matrix[route[b]][route[b + 1]]
                        new_len = route_lengths[i] + delta
                        if new_len < best_max:
                            new_max = max(new_len, max(route_lengths[k] for k in range(truck_count) if k != i))
                            if new_max < best_max:
                                route[a + 1:b + 1] = reversed(route[a + 1:b + 1])
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

        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    return best_routes