import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        total = 0.0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Initialization: each customer as a route
    route_list = [[0, c, 0] for c in customers]

    # Merge phase: greedily choose the merge that minimizes the new max distance
    while len(route_list) > truck_count:
        best_merge = None
        best_new_max = float('inf')
        for i in range(len(route_list)):
            for j in range(i+1, len(route_list)):
                ri = route_list[i]
                rj = route_list[j]
                # extract first and last customer (excluding depots)
                first_i = ri[1] if len(ri) > 2 else None
                last_i = ri[-2] if len(ri) > 2 else None
                first_j = rj[1] if len(rj) > 2 else None
                last_j = rj[-2] if len(rj) > 2 else None
                candidates = []
                if last_i is not None and first_j is not None:
                    new_route = ri[:-1] + rj[1:]
                    candidates.append(new_route)
                if last_j is not None and first_i is not None:
                    new_route = rj[:-1] + ri[1:]
                    candidates.append(new_route)
                for new_route in candidates:
                    # compute new max distance if merging i and j
                    new_max = 0.0
                    for idx, r in enumerate(route_list):
                        if idx == i or idx == j:
                            continue
                        d = route_distance(r)
                        if d > new_max:
                            new_max = d
                    d_new = route_distance(new_route)
                    if d_new > new_max:
                        new_max = d_new
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_merge = (i, j, new_route)
        if best_merge is None:
            # fallback: merge two smallest routes
            dists = [(route_distance(r), idx) for idx, r in enumerate(route_list)]
            dists.sort(key=lambda x: x[0])
            i = dists[0][1]
            j = dists[1][1]
            ri = route_list[i]
            rj = route_list[j]
            new_route = ri[:-1] + rj[1:]
            best_merge = (i, j, new_route)
        # apply merge
        i, j, new_route = best_merge
        new_route_list = []
        for idx, r in enumerate(route_list):
            if idx == i or idx == j:
                continue
            new_route_list.append(r)
        new_route_list.append(new_route)
        route_list = new_route_list
        report_best_vrp(route_list)

    # ensure exactly truck_count routes
    while len(route_list) > truck_count:
        dists = [(route_distance(r), idx) for idx, r in enumerate(route_list)]
        dists.sort(key=lambda x: x[0])
        i = dists[0][1]
        j = dists[1][1]
        ri = route_list[i]
        rj = route_list[j]
        new_route = ri[:-1] + rj[1:]
        new_route_list = []
        for idx, r in enumerate(route_list):
            if idx == i or idx == j:
                continue
            new_route_list.append(r)
        new_route_list.append(new_route)
        route_list = new_route_list

    report_best_vrp(route_list)

    # Adaptive improvement schedule (alternating relocate, swap, 2-opt)
    max_iter = max(200, len(customers) * truck_count)
    move_sequence = ['relocate', 'swap', '2opt'] * (max_iter // 3 + 1)
    for move_type in move_sequence[:max_iter]:
        improved = False
        dists = [route_distance(r) for r in route_list]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        max_route = route_list[max_idx]
        interior = max_route[1:-1]
        if not interior:
            break

        if move_type == 'relocate':
            for cust in interior:
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = route_list[other_idx]
                    best_pos = 0
                    best_delta = float('inf')
                    for pos in range(1, len(other_route)):
                        prev = other_route[pos-1]
                        nxt = other_route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = pos
                    new_routes = [list(r) for r in route_list]
                    new_routes[max_idx].remove(cust)
                    new_routes[other_idx].insert(best_pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_max - 1e-12:
                        route_list = new_routes
                        report_best_vrp(route_list)
                        improved = True
                        break
                if improved:
                    break
        elif move_type == 'swap':
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = route_list[other_idx]
                interior_other = other_route[1:-1]
                if not interior_other:
                    continue
                for cust_max in interior:
                    for cust_other in interior_other:
                        new_routes = [list(r) for r in route_list]
                        idx_max = new_routes[max_idx].index(cust_max)
                        idx_other = new_routes[other_idx].index(cust_other)
                        new_routes[max_idx][idx_max] = cust_other
                        new_routes[other_idx][idx_other] = cust_max
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max - 1e-12:
                            route_list = new_routes
                            report_best_vrp(route_list)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        elif move_type == '2opt':
            for idx in range(truck_count):
                route = route_list[idx]
                if len(route) <= 3:
                    continue
                best_route = route[:]
                best_dist = route_distance(route)
                found = False
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < best_dist - 1e-12:
                            best_dist = new_dist
                            best_route = new_route
                            found = True
                            break
                    if found:
                        break
                if found:
                    route_list[idx] = best_route
                    new_max = max(route_distance(r) for r in route_list)
                    if new_max < best_max - 1e-12:
                        report_best_vrp(route_list)
                    improved = True
                    break
        if not improved:
            continue

    final_routes = best_routes if best_routes is not None else route_list
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes