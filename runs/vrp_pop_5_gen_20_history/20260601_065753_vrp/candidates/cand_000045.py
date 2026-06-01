import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = set(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count

    def compute_route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def max_route_length():
        return max(route_lengths)

    report_best_max = float('inf')
    report_best_routes = None

    def report_best_vrp(routes):
        nonlocal report_best_max, report_best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < report_best_max:
            report_best_max = m
            report_best_routes = [list(r) for r in routes]

    # === Construction: cheapest insertion (minimizing max route length) ===
    while customers:
        best_cust = None
        best_new_max = float('inf')
        best_route = -1
        best_pos = -1
        for cust in customers:
            for r in range(truck_count):
                route = routes[r]
                for p in range(1, len(route)):
                    prev = route[p-1]
                    next_ = route[p]
                    old_edge = distance_matrix[prev, next_]
                    new_len = route_lengths[r] - old_edge + distance_matrix[prev, cust] + distance_matrix[cust, next_]
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != r and route_lengths[rr] > new_max:
                            new_max = route_lengths[rr]
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_cust = cust
                        best_route = r
                        best_pos = p
                    elif new_max == best_new_max and cust < best_cust:
                        best_new_max = new_max
                        best_cust = cust
                        best_route = r
                        best_pos = p
        if best_cust is None:
            break
        route = routes[best_route]
        route.insert(best_pos, best_cust)
        route_lengths[best_route] = compute_route_length(route)
        customers.remove(best_cust)

    current_max = max(route_lengths)
    report_best_vrp(routes)

    # === Improvement: best-accept local search ===
    max_iter = 5 * n
    for _ in range(max_iter):
        best_move = None
        best_new_max = current_max
        best_tie = None

        # Relocate moves
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust = route1[idx1]
                new_route1 = route1[:idx1] + route1[idx1+1:]
                len1_new = compute_route_length(new_route1)
                for t2 in range(truck_count):
                    if t1 == t2:
                        continue
                    route2 = routes[t2]
                    for pos in range(1, len(route2)):
                        new_route2 = route2[:pos] + [cust] + route2[pos:]
                        len2_new = compute_route_length(new_route2)
                        new_max = max(len1_new, len2_new)
                        for rr in range(truck_count):
                            if rr != t1 and rr != t2:
                                if route_lengths[rr] > new_max:
                                    new_max = route_lengths[rr]
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('relocate', t1, idx1, t2, pos)
                            best_tie = (t1, idx1, t2, pos)
                        elif new_max == best_new_max:
                            tie = (t1, idx1, t2, pos)
                            if best_tie is None or tie < best_tie:
                                best_new_max = new_max
                                best_move = ('relocate', t1, idx1, t2, pos)
                                best_tie = tie

        # Swap moves
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust1 = route1[idx1]
                for t2 in range(t1+1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    for idx2 in range(1, len(route2)-1):
                        cust2 = route2[idx2]
                        new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                        new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                        len1_new = compute_route_length(new_route1)
                        len2_new = compute_route_length(new_route2)
                        new_max = max(len1_new, len2_new)
                        for rr in range(truck_count):
                            if rr != t1 and rr != t2:
                                if route_lengths[rr] > new_max:
                                    new_max = route_lengths[rr]
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('swap', t1, idx1, t2, idx2)
                            best_tie = (t1, idx1, t2, idx2)
                        elif new_max == best_new_max:
                            tie = (t1, idx1, t2, idx2)
                            if best_tie is None or tie < best_tie:
                                best_new_max = new_max
                                best_move = ('swap', t1, idx1, t2, idx2)
                                best_tie = tie

        # 2-opt moves
        for t in range(truck_count):
            route = routes[t]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_len = compute_route_length(new_route)
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != t:
                            if route_lengths[rr] > new_max:
                                new_max = route_lengths[rr]
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = ('2opt', t, i, j)
                        best_tie = (t, i, j)
                    elif new_max == best_new_max:
                        tie = (t, i, j)
                        if best_tie is None or tie < best_tie:
                            best_new_max = new_max
                            best_move = ('2opt', t, i, j)
                            best_tie = tie

        if best_move is None or best_new_max >= current_max:
            break

        if best_move[0] == 'relocate':
            _, t1, idx1, t2, pos = best_move
            cust = routes[t1][idx1]
            del routes[t1][idx1]
            routes[t2].insert(pos, cust)
        elif best_move[0] == 'swap':
            _, t1, idx1, t2, idx2 = best_move
            cust1 = routes[t1][idx1]
            cust2 = routes[t2][idx2]
            routes[t1][idx1] = cust2
            routes[t2][idx2] = cust1
        else:  # '2opt'
            _, t, i, j = best_move
            routes[t] = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]

        route_lengths = [compute_route_length(r) for r in routes]
        current_max = max(route_lengths)
        report_best_vrp(routes)

    return report_best_routes if report_best_routes is not None else routes