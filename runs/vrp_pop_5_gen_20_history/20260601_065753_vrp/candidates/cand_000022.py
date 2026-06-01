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

    # === Regret-2 construction ===
    while customers:
        best_cust = None
        best_diff = -1.0
        best_route = -1
        best_pos = -1
        best_max = float('inf')
        for cust in customers:
            first_max = None
            second_max = None
            first_route = None
            first_pos = None
            for r in range(truck_count):
                route = routes[r]
                for p in range(1, len(route)):
                    prev_node = route[p-1]
                    next_node = route[p]
                    old_edge = distance_matrix[prev_node, next_node]
                    new_len = route_lengths[r] - old_edge + distance_matrix[prev_node, cust] + distance_matrix[cust, next_node]
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != r and route_lengths[rr] > new_max:
                            new_max = route_lengths[rr]
                    if first_max is None or new_max < first_max:
                        second_max = first_max
                        first_max = new_max
                        first_route = r
                        first_pos = p
                    elif second_max is None or new_max < second_max:
                        second_max = new_max
            if first_max is None:
                continue
            if second_max is None:
                diff = float('inf')
            else:
                diff = first_max - second_max
            if diff > best_diff:
                best_diff = diff
                best_cust = cust
                best_route = first_route
                best_pos = first_pos
                best_max = first_max
            elif diff == best_diff and cust < best_cust:
                best_cust = cust
                best_route = first_route
                best_pos = first_pos
                best_max = first_max
        # Insert best customer
        route = routes[best_route]
        route.insert(best_pos, best_cust)
        route_lengths[best_route] = compute_route_length(route)
        customers.remove(best_cust)

    current_max = max_route_length()
    best_max = current_max
    best_routes = [list(r) for r in routes]

    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]

    report_best_vrp(routes)

    # === Phase 1: Best-accept local search ===
    max_iter = 2 * n
    for _ in range(max_iter):
        improved = False
        best_move = None
        best_new_max = current_max
        best_tie = None

        # Relocate moves
        for t1 in range(truck_count):
            if len(routes[t1]) <= 2:
                continue
            for idx1 in range(1, len(routes[t1])-1):
                cust = routes[t1][idx1]
                new_route1 = routes[t1][:idx1] + routes[t1][idx1+1:]
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
            if len(routes[t1]) <= 2:
                continue
            for idx1 in range(1, len(routes[t1])-1):
                cust1 = routes[t1][idx1]
                for t2 in range(t1+1, truck_count):
                    if len(routes[t2]) <= 2:
                        continue
                    for idx2 in range(1, len(routes[t2])-1):
                        cust2 = routes[t2][idx2]
                        new_route1 = routes[t1][:idx1] + [cust2] + routes[t1][idx1+1:]
                        new_route2 = routes[t2][:idx2] + [cust1] + routes[t2][idx2+1:]
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
            if len(routes[t]) <= 3:
                continue
            for i in range(1, len(routes[t])-2):
                for j in range(i+1, len(routes[t])-1):
                    new_route = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
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

        if best_move is not None and best_new_max < current_max:
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
            elif best_move[0] == '2opt':
                _, t, i, j = best_move
                routes[t] = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
            route_lengths = [compute_route_length(r) for r in routes]
            current_max = max(route_lengths)
            report_best_vrp(routes)
            improved = True
        if not improved:
            break

    # === Phase 2: Tabu search ===
    tabu_tenure = 7
    tabu_list = []
    tabu_set = set()
    max_iter_tabu = n  # finite bound
    for _ in range(max_iter_tabu):
        best_move = None
        best_new_max = float('inf')
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
                        # Tabu check: reverse move (cust, t2, t1)
                        is_tabu = (cust, t2, t1) in tabu_set
                        if is_tabu and new_max >= best_max:
                            continue
                        tie = (new_max, 0, t1, idx1, t2, pos)
                        if best_tie is None or tie < best_tie:
                            best_new_max = new_max
                            best_move = ('relocate', t1, idx1, t2, pos, cust)
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
                        # Tabu check
                        tabu1 = (cust1, cust2, t1, t2)
                        tabu2 = (cust2, cust1, t2, t1)
                        is_tabu = (tabu1 in tabu_set) or (tabu2 in tabu_set)
                        if is_tabu and new_max >= best_max:
                            continue
                        tie = (new_max, 1, t1, idx1, t2, idx2)
                        if best_tie is None or tie < best_tie:
                            best_new_max = new_max
                            best_move = ('swap', t1, idx1, t2, idx2, cust1, cust2)
                            best_tie = tie

        # 2-opt moves (intra-route, not tabu)
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
                    tie = (new_max, 2, t, i, j, -1, -1)
                    if best_tie is None or tie < best_tie:
                        best_new_max = new_max
                        best_move = ('2opt', t, i, j, new_route)
                        best_tie = tie

        if best_move is None:
            break

        # Apply best move
        if best_move[0] == 'relocate':
            _, t1, idx1, t2, pos, cust = best_move
            routes[t1] = routes[t1][:idx1] + routes[t1][idx1+1:]
            routes[t2] = routes[t2][:pos] + [cust] + routes[t2][pos:]
            route_lengths[t1] = compute_route_length(routes[t1])
            route_lengths[t2] = compute_route_length(routes[t2])
            tabu_entry = (cust, t2, t1)
            tabu_list.append(tabu_entry)
            tabu_set.add(tabu_entry)
        elif best_move[0] == 'swap':
            _, t1, idx1, t2, idx2, cust1, cust2 = best_move
            routes[t1][idx1] = cust2
            routes[t2][idx2] = cust1
            route_lengths[t1] = compute_route_length(routes[t1])
            route_lengths[t2] = compute_route_length(routes[t2])
            tabu_entry1 = (cust1, cust2, t1, t2)
            tabu_entry2 = (cust2, cust1, t2, t1)
            tabu_list.append(tabu_entry1)
            tabu_set.add(tabu_entry1)
            tabu_list.append(tabu_entry2)
            tabu_set.add(tabu_entry2)
        else:  # '2opt'
            _, t, i, j, new_route = best_move
            routes[t] = new_route
            route_lengths[t] = compute_route_length(new_route)

        # Manage tabu list
        while len(tabu_list) > tabu_tenure:
            entry = tabu_list.pop(0)
            tabu_set.discard(entry)

        current_max = max(route_lengths)
        report_best_vrp(routes)

    return best_routes