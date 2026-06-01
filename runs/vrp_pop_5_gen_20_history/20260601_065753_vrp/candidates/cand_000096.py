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

    def decode(perm):
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
        max_len = max(lengths)
        return routes, lengths, max_len

    # initial solution from random permutation
    perm = customers[:]
    random.shuffle(perm)
    routes, lengths, current_max = decode(perm)
    report_best_vrp(routes)

    max_iter = 5 * n  # bounded loop
    tabu_tenure = 10
    tabu_list = {}  # map attribute -> iteration when becomes non-tabu
    current_iter = 0

    for iteration in range(max_iter):
        current_iter = iteration
        # identify longest routes
        max_len = max(lengths)
        candidates = [i for i, l in enumerate(lengths) if abs(l - max_len) < 1e-12]

        moves = []  # each move is (type, t1, i, t2, j, cust, new_max) or (type, t, i, j, new_max)

        # relocate moves from longest routes
        for t1 in candidates:
            route1 = routes[t1]
            for i in range(1, len(route1)-1):
                cust = route1[i]
                for t2 in range(truck_count):
                    if t2 == t1:
                        continue
                    route2 = routes[t2]
                    for j in range(1, len(route2)-1):
                        # compute new lengths
                        new_len1 = lengths[t1] - distance_matrix[route1[i-1], route1[i]] - distance_matrix[route1[i], route1[i+1]] + distance_matrix[route1[i-1], route1[i+1]]
                        new_len2 = lengths[t2] - distance_matrix[route2[j-1], route2[j]] + distance_matrix[route2[j-1], cust] + distance_matrix[cust, route2[j]]
                        new_max_local = new_len1
                        for rr in range(truck_count):
                            if rr == t1:
                                if new_len1 > new_max_local: new_max_local = new_len1
                            elif rr == t2:
                                if new_len2 > new_max_local: new_max_local = new_len2
                            else:
                                if lengths[rr] > new_max_local: new_max_local = lengths[rr]
                        moves.append(('relocate', t1, i, t2, j, cust, new_max_local))

        # 2-opt moves on longest routes
        for t in candidates:
            route = routes[t]
            L = len(route)
            for i in range(1, L-2):
                for j in range(i+1, L-1):
                    new_len = lengths[t] - distance_matrix[route[i-1], route[i]] - distance_matrix[route[j], route[j+1]] + distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    new_max_local = new_len
                    for rr in range(truck_count):
                        if rr != t and lengths[rr] > new_max_local:
                            new_max_local = lengths[rr]
                    moves.append(('2opt', t, i, j, new_max_local))

        if not moves:
            # diversification: randomly perturb by moving a customer from a longest route
            max_len = max(lengths)
            candidates = [i for i, l in enumerate(lengths) if abs(l - max_len) < 1e-12]
            t1 = random.choice(candidates)
            route1 = routes[t1]
            if len(route1) > 2:
                i = random.randint(1, len(route1)-2)
                cust = route1[i]
                t2 = random.randint(0, truck_count-1)
                if t2 != t1:
                    j = random.randint(1, len(routes[t2])-1)
                    routes[t1].pop(i)
                    routes[t2].insert(j, cust)
                    lengths[t1] = route_length(routes[t1])
                    lengths[t2] = route_length(routes[t2])
                    report_best_vrp(routes)
            continue

        # sort moves by new_max, then type (relocate before 2opt), then t1, i, etc.
        moves.sort(key=lambda x: (x[-1], 0 if x[0]=='relocate' else 1, x[1], x[2], x[3] if len(x)>4 else 0))

        best_move = None
        for m in moves:
            # aspiration: if new max improves best, accept
            if m[-1] < best_max - 1e-12:
                best_move = m
                break
            # tabu check
            if m[0] == 'relocate':
                tabu_key = ('relocate', m[5], m[1])  # customer and from_route
            else:
                tabu_key = ('2opt', m[1], m[2], m[3])
            if tabu_key not in tabu_list or tabu_list[tabu_key] <= current_iter:
                best_move = m
                break

        if best_move is None:
            # no non-tabu improving move; diversify by random perturbation
            max_len = max(lengths)
            candidates = [i for i, l in enumerate(lengths) if abs(l - max_len) < 1e-12]
            t1 = random.choice(candidates)
            route1 = routes[t1]
            if len(route1) > 2:
                i = random.randint(1, len(route1)-2)
                cust = route1[i]
                t2 = random.randint(0, truck_count-1)
                if t2 != t1:
                    j = random.randint(1, len(routes[t2])-1)
                    routes[t1].pop(i)
                    routes[t2].insert(j, cust)
                    lengths[t1] = route_length(routes[t1])
                    lengths[t2] = route_length(routes[t2])
                    report_best_vrp(routes)
            continue

        # apply best_move
        if best_move[0] == 'relocate':
            _, t1, i, t2, j, cust, new_max = best_move
            routes[t1].pop(i)
            routes[t2].insert(j, cust)
            lengths[t1] = route_length(routes[t1])
            lengths[t2] = route_length(routes[t2])
            # add tabu for the reverse move (moving cust back to t1)
            tabu_list[('relocate', cust, t1)] = current_iter + tabu_tenure
        else:
            _, t, i, j, new_max = best_move
            route = routes[t]
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            routes[t] = new_route
            lengths[t] = route_length(new_route)
            # add tabu for reversing same segment
            tabu_list[('2opt', t, i, j)] = current_iter + tabu_tenure

        # update current max and best
        current_max = max(lengths)
        if current_max < best_max - 1e-12:
            report_best_vrp(routes)

    return best_routes