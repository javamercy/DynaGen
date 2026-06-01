import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]

    def compute_route_length(route):
        length = 0.0
        for i in range(len(route) - 1):
            length += distance_matrix[route[i], route[i+1]]
        return length

    # ---------- min-max insertion construction ----------
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count

    for cust in customers:
        best_max = float('inf')
        best_route = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            for p in range(1, len(route)):
                prev = route[p-1]
                nxt = route[p]
                old_edge = distance_matrix[prev, nxt]
                new_len = route_lengths[r] - old_edge + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                new_max = new_len
                for rr in range(truck_count):
                    if rr != r and route_lengths[rr] > new_max:
                        new_max = route_lengths[rr]
                if new_max < best_max or (new_max == best_max and (r < best_route or (r == best_route and p < best_pos))):
                    best_max = new_max
                    best_route = r
                    best_pos = p
        routes[best_route].insert(best_pos, cust)
        route_lengths[best_route] = compute_route_length(routes[best_route])

    current_max = max(route_lengths)
    best_max = current_max
    best_routes = [list(r) for r in routes]

    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]

    report_best_vrp(routes)

    # ---------- local search (best improvement) ----------
    def local_search(routes, route_lengths):
        current_max = max(route_lengths)
        max_iter = 2 * n
        for _ in range(max_iter):
            improved = False
            best_move = None
            best_new_max = current_max
            best_tie = None

            # relocate
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
                            if new_max < best_new_max or (new_max == best_new_max and (t1, idx1, t2, pos) < best_tie):
                                best_new_max = new_max
                                best_move = ('relocate', t1, idx1, t2, pos)
                                best_tie = (t1, idx1, t2, pos)

            # swap
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
                            if new_max < best_new_max or (new_max == best_new_max and (t1, idx1, t2, idx2) < best_tie):
                                best_new_max = new_max
                                best_move = ('swap', t1, idx1, t2, idx2)
                                best_tie = (t1, idx1, t2, idx2)

            # 2-opt
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
                        if new_max < best_new_max or (new_max == best_new_max and (t, i, j) < best_tie):
                            best_new_max = new_max
                            best_move = ('2opt', t, i, j)
                            best_tie = (t, i, j)

            if best_move is not None and best_new_max < current_max:
                improved = True
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
                else:
                    _, t, i, j = best_move
                    routes[t] = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
                route_lengths = [compute_route_length(r) for r in routes]
                current_max = max(route_lengths)
                report_best_vrp(routes)
            if not improved:
                break
        return routes, route_lengths

    # ---------- destroy and repair ----------
    def worst_removal(routes, route_lengths, n_remove):
        # compute contribution for each customer (excl depots)
        contributions = []
        for r in range(truck_count):
            route = routes[r]
            for idx in range(1, len(route)-1):
                cust = route[idx]
                prev = route[idx-1]
                nxt = route[idx+1]
                contrib = distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                contributions.append((contrib, cust, r, idx))
        # sort by contribution descending, then customer index ascending for tie
        contributions.sort(key=lambda x: (-x[0], x[1]))
        to_remove = contributions[:n_remove]
        # remove in reverse order of index to keep indices valid
        to_remove.sort(key=lambda x: -x[3])
        removed_customers = []
        for _, cust, r, idx in to_remove:
            removed_customers.append((cust, r, idx))
        # actually remove
        for cust, r, idx in removed_customers:
            del routes[r][idx]
        route_lengths = [compute_route_length(r) for r in routes]
        # return removed customers (sorted by contribution descending then index)
        removed_sorted = [x[1] for x in contributions[:n_remove]]
        return removed_sorted, routes, route_lengths

    def repair(routes, route_lengths, removed_customers):
        for cust in removed_customers:
            best_max = float('inf')
            best_route = -1
            best_pos = -1
            for r in range(truck_count):
                route = routes[r]
                for p in range(1, len(route)):
                    prev = route[p-1]
                    nxt = route[p]
                    old_edge = distance_matrix[prev, nxt]
                    new_len = route_lengths[r] - old_edge + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != r and route_lengths[rr] > new_max:
                            new_max = route_lengths[rr]
                    if new_max < best_max or (new_max == best_max and (r < best_route or (r == best_route and p < best_pos))):
                        best_max = new_max
                        best_route = r
                        best_pos = p
            routes[best_route].insert(best_pos, cust)
            route_lengths[best_route] = compute_route_length(routes[best_route])
        return routes, route_lengths

    # ---------- main loop ----------
    max_outer = (n - 1) * truck_count * 2
    for iteration in range(max_outer):
        n_remove = max(1, int(n * 0.2))
        # destroy
        removed, routes, route_lengths = worst_removal(routes, route_lengths, n_remove)
        # repair
        routes, route_lengths = repair(routes, route_lengths, removed)
        # local search
        routes, route_lengths = local_search(routes, route_lengths)
        current_max = max(route_lengths)
        # always accept (replace current)
        # best is updated via report_best_vrp inside local_search

    return best_routes