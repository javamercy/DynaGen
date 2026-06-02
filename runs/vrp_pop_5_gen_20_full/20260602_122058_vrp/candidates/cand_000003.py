import numpy as np
import heapq
from itertools import chain

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count
    unassigned = set(customers)

    # Precompute all insertion costs? We'll compute on fly.
    # Insertion cost for placing customer c into route r at position p (0..len(r)-1)
    def insertion_cost(r, pos, c):
        prev = r[pos]
        nxt = r[pos+1]
        return distance_matrix[prev][c] + distance_matrix[c][nxt] - distance_matrix[prev][nxt]

    # Best insertion for customer c into all routes, returns list of (cost, route_idx, pos)
    def best_insertions(c):
        best = None
        second_best = None
        best_info = None
        for r_idx, r in enumerate(routes):
            for pos in range(len(r)-1):
                cost = insertion_cost(r, pos, c)
                if best is None or cost < best:
                    second_best = best
                    best = cost
                    best_info = (r_idx, pos)
                elif second_best is None or cost < second_best:
                    second_best = cost
        if second_best is None:
            second_best = best  # only one route feasible
        return best, second_best, best_info

    best_max_dist = float('inf')
    best_routes = None

    def compute_max_dist():
        return max(route_lengths)

    def update_best():
        nonlocal best_max_dist, best_routes
        m = compute_max_dist()
        if m < best_max_dist - 1e-9:
            best_max_dist = m
            best_routes = [list(route) for route in routes]
            from code import report_best_vrp
            report_best_vrp(best_routes)

    while unassigned:
        max_regret = -1.0
        best_customer = None
        best_route_idx = None
        best_pos = None
        best_delta = None
        for c in unassigned:
            delta, second_delta, info = best_insertions(c)
            regret = second_delta - delta
            # tie-breaking: larger regret, then smaller customer id
            if (regret > max_regret) or (abs(regret - max_regret) < 1e-9 and (best_customer is None or c < best_customer)):
                max_regret = regret
                best_customer = c
                best_delta = delta
                best_route_idx, best_pos = info
        # Insert best_customer
        r_idx = best_route_idx
        pos = best_pos
        c = best_customer
        routes[r_idx].insert(pos+1, c)
        route_lengths[r_idx] += best_delta
        unassigned.remove(c)
        update_best()

    # Improvement: inter-route relocate (move one customer) to reduce max distance
    # Bounded loops: at most (n * truck_count) iterations per phase
    max_iter = n * truck_count
    for _ in range(max_iter):
        # Find route with max length
        max_len = max(route_lengths)
        max_idx = route_lengths.index(max_len)
        # Try to move a customer from max_idx to another route
        better = False
        route_len_before = route_lengths[max_idx]
        # iterate over customers in longest route (excluding depot)
        custs = routes[max_idx][1:-1]
        if not custs:
            break
        for c in custs:
            # Remove c from its current route
            cur_route = routes[max_idx]
            # Find position of c
            pos_c = cur_route.index(c)
            # Calculate removal savings
            prev_c = cur_route[pos_c-1]
            nxt_c = cur_route[pos_c+1]
            saving = distance_matrix[prev_c][c] + distance_matrix[c][nxt_c] - distance_matrix[prev_c][nxt_c]
            # Try to insert into other routes
            for r_idx in range(truck_count):
                if r_idx == max_idx:
                    continue
                other_route = routes[r_idx]
                for pos in range(len(other_route)-1):
                    cost = insertion_cost(r_idx, pos, c)
                    new_route_lengths = route_lengths[:]
                    new_route_lengths[max_idx] -= saving
                    new_route_lengths[r_idx] += cost
                    new_max = max(new_route_lengths)
                    if new_max < best_max_dist - 1e-9:
                        # Accept move
                        # Remove c from old route
                        cur_route.pop(pos_c)
                        route_lengths[max_idx] -= saving
                        # Insert into new route
                        other_route.insert(pos+1, c)
                        route_lengths[r_idx] += cost
                        better = True
                        update_best()
                        break
                if better:
                    break
            if better:
                break
        if not better:
            break

    # Intra-route 2-opt for each route (bounded)
    for r_idx in range(truck_count):
        route = routes[r_idx]
        improved = True
        iters = 0
        while improved and iters < len(route):
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # swap edges (i-1,i) and (j,j+1) with (i-1,j) and (i,j+1)
                    a = route[i-1]
                    b = route[i]
                    c = route[j]
                    d = route[j+1]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new = distance_matrix[a][c] + distance_matrix[b][d]
                    if new < old - 1e-9:
                        # reverse segment i..j
                        route[i:j+1] = reversed(route[i:j+1])
                        route_lengths[r_idx] += new - old
                        improved = True
                        update_best()
                        break
                if improved:
                    break
            iters += 1

    return best_routes if best_routes is not None else routes