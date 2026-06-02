import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    customers = list(range(1, n))
    best_routes = None
    best_max = float('inf')
    num_restarts = 5
    for restart in range(num_restarts):
        random.seed(restart)
        shuffled = list(customers)
        random.shuffle(shuffled)
        routes = [[0, 0] for _ in range(truck_count)]
        route_lengths = [0.0] * truck_count
        for cust in shuffled:
            best_max_local = float('inf')
            best_route = None
            best_pos = None
            for ri, route in enumerate(routes):
                cur_len = route_lengths[ri]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    add = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                    new_len = cur_len + add
                    new_max = max(route_lengths[:ri] + [new_len] + route_lengths[ri+1:])
                    if new_max < best_max_local or (new_max == best_max_local and new_len < cur_len):
                        best_max_local = new_max
                        best_route = ri
                        best_pos = pos
            route = routes[best_route]
            prev = route[best_pos-1]
            nxt = route[best_pos]
            route.insert(best_pos, cust)
            route_lengths[best_route] += distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        # Best-improvement local search
        max_iterations = 1000  # safe bound
        for _ in range(max_iterations):
            best_move = None
            best_new_max = None
            best_delta_info = None
            # Evaluate all inter-route relocate moves
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
                            if new_max < current_max:
                                if best_move is None or new_max < best_new_max:
                                    best_move = ('relocate', i, pos_i, j, pos_j)
                                    best_new_max = new_max
                                    best_delta_info = (removal_delta, insert_delta)
            # Evaluate all inter-route swap moves
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
                            if new_max < current_max:
                                if best_move is None or new_max < best_new_max:
                                    best_move = ('swap', i, pos_i, j, pos_j)
                                    best_new_max = new_max
                                    best_delta_info = (delta_i_rem, delta_j_rem, add_i, add_j)
            # Evaluate all intra-route 2-opt moves
            for i in range(truck_count):
                route = routes[i]
                if len(route) <= 3:
                    continue
                for a in range(0, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        delta = distance_matrix[route[a]][route[b]] + distance_matrix[route[a+1]][route[b+1]] - distance_matrix[route[a]][route[a+1]] - distance_matrix[route[b]][route[b+1]]
                        new_len = route_lengths[i] + delta
                        new_max = max(new_len, max(route_lengths[k] for k in range(truck_count) if k != i))
                        if new_max < current_max:
                            if best_move is None or new_max < best_new_max:
                                best_move = ('2-opt', i, a, b)
                                best_new_max = new_max
                                best_delta_info = delta
            if best_move is None:
                break
            # Apply best move
            if best_move[0] == 'relocate':
                _, i, pos_i, j, pos_j = best_move
                cust = routes[i].pop(pos_i)
                routes[j].insert(pos_j, cust)
                route_lengths[i] += best_delta_info[0]  # removal_delta
                route_lengths[j] += best_delta_info[1]  # insert_delta
            elif best_move[0] == 'swap':
                _, i, pos_i, j, pos_j = best_move
                cust_i = routes[i][pos_i]
                cust_j = routes[j][pos_j]
                routes[i][pos_i] = cust_j
                routes[j][pos_j] = cust_i
                delta_i_rem, delta_j_rem, add_i, add_j = best_delta_info
                route_lengths[i] += delta_i_rem + add_i
                route_lengths[j] += delta_j_rem + add_j
            elif best_move[0] == '2-opt':
                _, i, a, b = best_move
                route = routes[i]
                route[a+1:b+1] = reversed(route[a+1:b+1])
                route_lengths[i] += best_delta_info
            current_max = best_new_max
            if current_max < best_max:
                best_max = current_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
    return best_routes