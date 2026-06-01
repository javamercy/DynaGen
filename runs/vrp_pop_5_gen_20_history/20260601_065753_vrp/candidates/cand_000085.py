import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def compute_route_length(route):
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
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]

    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        for cust in perm:
            best_max_candidate = float('inf')
            best_r = -1
            best_p = -1
            best_total_inc = float('inf')
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
                    total_inc = new_len - lengths[r]
                    if new_max < best_max_candidate or (new_max == best_max_candidate and total_inc < best_total_inc):
                        best_max_candidate = new_max
                        best_r = r
                        best_p = p
                        best_total_inc = total_inc
            routes[best_r].insert(best_p, cust)
            lengths[best_r] = compute_route_length(routes[best_r])
        max_len = max(lengths)
        return routes, lengths, max_len

    def local_search(routes, lengths):
        max_iter = min(200, 10 * (n - 1))
        current_max = max(lengths)
        for _ in range(max_iter):
            move_type = random.randint(0, 2)
            best_new_max = current_max
            best_move = None
            if move_type == 0:  # relocate
                t1 = random.randint(0, truck_count - 1)
                if len(routes[t1]) <= 2:
                    continue
                i = random.randint(1, len(routes[t1]) - 2)
                t2 = random.randint(0, truck_count - 1)
                if t2 == t1:
                    continue
                j = random.randint(1, len(routes[t2]) - 1)
                cust = routes[t1][i]
                new_route1 = routes[t1][:i] + routes[t1][i+1:]
                new_len1 = compute_route_length(new_route1)
                new_route2 = routes[t2][:j] + [cust] + routes[t2][j:]
                new_len2 = compute_route_length(new_route2)
                new_max = max(new_len1, new_len2)
                for k in range(truck_count):
                    if k not in (t1, t2):
                        new_max = max(new_max, lengths[k])
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = ('relocate', t1, i, t2, j, cust)
            elif move_type == 1:  # swap
                t1 = random.randint(0, truck_count - 1)
                if len(routes[t1]) <= 2:
                    continue
                i = random.randint(1, len(routes[t1]) - 2)
                t2 = random.randint(0, truck_count - 1)
                if t2 == t1 or len(routes[t2]) <= 2:
                    continue
                j = random.randint(1, len(routes[t2]) - 2)
                cust1 = routes[t1][i]
                cust2 = routes[t2][j]
                new_route1 = routes[t1][:i] + [cust2] + routes[t1][i+1:]
                new_route2 = routes[t2][:j] + [cust1] + routes[t2][j+1:]
                new_len1 = compute_route_length(new_route1)
                new_len2 = compute_route_length(new_route2)
                new_max = max(new_len1, new_len2)
                for k in range(truck_count):
                    if k not in (t1, t2):
                        new_max = max(new_max, lengths[k])
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = ('swap', t1, i, t2, j, cust1, cust2)
            else:  # 2-opt
                t = random.randint(0, truck_count - 1)
                if len(routes[t]) <= 3:
                    continue
                i = random.randint(1, len(routes[t]) - 3)
                j = random.randint(i + 1, len(routes[t]) - 2)
                new_route = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
                new_len = compute_route_length(new_route)
                new_max = new_len
                for k in range(truck_count):
                    if k != t:
                        new_max = max(new_max, lengths[k])
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = ('2opt', t, i, j, new_route)
            if best_move is not None:
                if best_move[0] == 'relocate':
                    _, t1, i, t2, j, cust = best_move
                    routes[t1].pop(i)
                    routes[t2].insert(j, cust)
                    lengths[t1] = compute_route_length(routes[t1])
                    lengths[t2] = compute_route_length(routes[t2])
                elif best_move[0] == 'swap':
                    _, t1, i, t2, j, cust1, cust2 = best_move
                    routes[t1][i] = cust2
                    routes[t2][j] = cust1
                    lengths[t1] = compute_route_length(routes[t1])
                    lengths[t2] = compute_route_length(routes[t2])
                else:
                    _, t, i, j, new_route = best_move
                    routes[t] = new_route
                    lengths[t] = new_len
                current_max = max(lengths)
        return routes, lengths

    num_restarts = min(50, n)
    for _ in range(num_restarts):
        perm = customers[:]
        random.shuffle(perm)
        routes, lengths, max_len = decode(perm)
        report_best_vrp(routes)
        routes, lengths = local_search(routes, lengths)
        max_len = max(lengths)
        if max_len < best_max:
            report_best_vrp(routes)

    return best_routes