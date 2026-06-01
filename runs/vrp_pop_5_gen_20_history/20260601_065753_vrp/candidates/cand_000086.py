import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_length(route):
        if len(route) <= 1:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_max = float('inf')
    best_routes = None

    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(route_length(r) for r in routes)
        if m < best_max - 1e-12:
            best_max = m
            best_routes = [list(r) for r in routes]

    def greedy_insertion(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        for cust in perm:
            best_max_val = float('inf')
            best_inc = float('inf')
            best_r = -1
            best_p = -1
            for r in range(truck_count):
                route = routes[r]
                for p in range(1, len(route)):
                    prev = route[p-1]
                    nxt = route[p]
                    new_len = lengths[r] - distance_matrix[prev, nxt] + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != r and lengths[rr] > new_max:
                            new_max = lengths[rr]
                    inc = new_len - lengths[r]
                    if new_max < best_max_val or (abs(new_max - best_max_val) < 1e-12 and inc < best_inc):
                        best_max_val = new_max
                        best_inc = inc
                        best_r = r
                        best_p = p
            routes[best_r].insert(best_p, cust)
            lengths[best_r] = route_length(routes[best_r])
        return routes, lengths

    random.seed(0)
    perm = customers[:]
    random.shuffle(perm)
    routes, lengths = greedy_insertion(perm)
    max_len = max(lengths)
    report_best_vrp(routes)
    current_routes = [list(r) for r in routes]
    current_lengths = lengths[:]
    current_max = max_len

    max_iter = 5 * n
    tabu_tenure = n // 2 + 1
    tabu_list = {}

    def evaluate_moves(routes, lengths):
        candidates = []
        n_routes = truck_count
        # relocate moves
        for r1 in range(n_routes):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for i in range(1, len(route1)-1):
                cust = route1[i]
                for r2 in range(n_routes):
                    if r2 == r1:
                        continue
                    route2 = routes[r2]
                    for j in range(1, len(route2)):
                        new_len1 = lengths[r1] - distance_matrix[route1[i-1], cust] - distance_matrix[cust, route1[i+1]] + distance_matrix[route1[i-1], route1[i+1]]
                        new_len2 = lengths[r2] + distance_matrix[route2[j-1], cust] + distance_matrix[cust, route2[j]] - distance_matrix[route2[j-1], route2[j]]
                        new_max = max(new_len1, new_len2, *[lengths[rr] for rr in range(n_routes) if rr not in (r1, r2)])
                        move_details = ('relocate', r1, i, r2, j, cust, new_len1, new_len2)
                        tie_key = (new_max, 0, r1, i, r2, j)
                        tabu_key = (cust, r1, r2)
                        candidates.append((new_max, tie_key, move_details, tabu_key))
        # 2-opt moves
        for r in range(n_routes):
            route = routes[r]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_len = lengths[r]
                    new_len += distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]] - distance_matrix[route[i-1], route[i]] - distance_matrix[route[j], route[j+1]]
                    new_max = max(new_len, *[lengths[rr] for rr in range(n_routes) if rr != r])
                    move_details = ('2opt', r, i, j, None)
                    tie_key = (new_max, 1, r, i, j)
                    tabu_key = (r, i, j)
                    candidates.append((new_max, tie_key, move_details, tabu_key))
        return candidates

    for iteration in range(max_iter):
        candidates = evaluate_moves(current_routes, current_lengths)
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1])
        best_candidate = None
        for candidate in candidates:
            new_max, _, move_details, tabu_key = candidate
            tabu_exp = tabu_list.get(tabu_key, -1)
            if iteration < tabu_exp:
                if new_max < best_max - 1e-12:
                    best_candidate = candidate
                    break
            else:
                best_candidate = candidate
                break
        if best_candidate is None:
            break
        new_max, _, move_details, tabu_key = best_candidate
        if move_details[0] == 'relocate':
            _, r1, i, r2, j, cust, new_len1, new_len2 = move_details
            current_routes[r1].pop(i)
            current_routes[r2].insert(j, cust)
            current_lengths[r1] = new_len1
            current_lengths[r2] = new_len2
        else:
            _, r, i, j, _ = move_details
            route = current_routes[r]
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            current_routes[r] = new_route
            current_lengths[r] = route_length(new_route)
        current_max = max(current_lengths)
        tabu_list[tabu_key] = iteration + tabu_tenure
        if current_max < best_max - 1e-12:
            report_best_vrp(current_routes)

    return best_routes