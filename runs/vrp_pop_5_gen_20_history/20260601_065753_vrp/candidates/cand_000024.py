import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count

    def compute_route_length(route):
        length = 0.0
        for i in range(len(route) - 1):
            length += distance_matrix[route[i], route[i+1]]
        return length

    # Construction: regret-2 min-max insertion
    unvisited = set(customers)
    while unvisited:
        best_cust = None
        best_regret = -1.0
        best_route_idx = -1
        best_pos = -1
        best_max = float('inf')
        for cust in list(unvisited):
            # Evaluate insertion in each route
            insertions = []
            for r in range(truck_count):
                route = routes[r]
                for p in range(1, len(route)):
                    prev_node = route[p-1]
                    next_node = route[p]
                    old_edge = distance_matrix[prev_node, next_node]
                    new_len = route_lengths[r] - old_edge + distance_matrix[prev_node, cust] + distance_matrix[cust, next_node]
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != r:
                            if route_lengths[rr] > new_max:
                                new_max = route_lengths[rr]
                    insertions.append((new_max, r, p))
            # Sort by new_max ascending
            insertions.sort(key=lambda x: (x[0], x[1], x[2]))
            if len(insertions) >= 2:
                regret = insertions[1][0] - insertions[0][0]
            else:
                regret = 0.0
            # Choose customer with highest regret; tie-break on best max
            if regret > best_regret or (regret == best_regret and insertions[0][0] < best_max):
                best_regret = regret
                best_cust = cust
                best_max = insertions[0][0]
                best_route_idx = insertions[0][1]
                best_pos = insertions[0][2]
            elif regret == best_regret and insertions[0][0] == best_max:
                # Tie: smaller route idx, then smaller position
                if insertions[0][1] < best_route_idx or (insertions[0][1] == best_route_idx and insertions[0][2] < best_pos):
                    best_cust = cust
                    best_route_idx = insertions[0][1]
                    best_pos = insertions[0][2]
        # Insert best customer
        route = routes[best_route_idx]
        route.insert(best_pos, best_cust)
        route_lengths[best_route_idx] = compute_route_length(route)
        unvisited.remove(best_cust)

    current_max = max(route_lengths)
    best_max = current_max
    best_routes = [list(r) for r in routes]
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        max_cost = max(compute_route_length(r) for r in routes)
        if max_cost < best_max:
            best_max = max_cost
            best_routes = [list(r) for r in routes]
    report_best_vrp(routes)

    # Local search with patience
    max_iter = 2 * n
    patience = n  # consecutive non-improving moves allowed
    no_improve_count = 0
    for _ in range(max_iter):
        improved = False
        best_move = None
        best_new_max = current_max
        best_tie = None

        # Relocate moves (inter-route)
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

        # Swap moves (inter-route)
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

        # 2-opt moves (intra-route)
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

        if best_move is not None and best_new_max < current_max:
            # Apply best move
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
            # Update route lengths
            route_lengths = [compute_route_length(r) for r in routes]
            current_max = max(route_lengths)
            report_best_vrp(routes)
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

    return best_routes