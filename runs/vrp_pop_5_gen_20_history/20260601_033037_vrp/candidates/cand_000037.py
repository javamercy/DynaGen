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
        for i in range(len(route)-1):
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

    # Clarke-Wright savings construction (deterministic)
    route_list = [[0, c, 0] for c in customers]
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))

    cust_to_route = {c: idx for idx, r in enumerate(route_list) for c in r[1:-1]}
    endpoints = []
    for idx, r in enumerate(route_list):
        interior = r[1:-1]
        if interior:
            first = interior[0]
            last = interior[-1]
        else:
            first = last = None
        endpoints.append((first, last, idx))

    for _, i, j in savings:
        if len(route_list) <= truck_count:
            break
        if i not in cust_to_route or j not in cust_to_route:
            continue
        ri = cust_to_route[i]
        rj = cust_to_route[j]
        if ri == rj:
            continue
        first_i, last_i, _ = endpoints[ri]
        first_j, last_j, _ = endpoints[rj]
        merged = None
        if i == last_i and j == first_j:
            merged = route_list[ri][:-1] + route_list[rj][1:]
        elif j == last_j and i == first_i:
            merged = route_list[rj][:-1] + route_list[ri][1:]
        elif i == first_i and j == last_j:
            merged = route_list[rj][:-1] + route_list[ri][1:]
        elif j == first_j and i == last_i:
            merged = route_list[ri][:-1] + route_list[rj][1:]
        else:
            continue
        if merged is None:
            continue
        new_route_list = []
        for idx2, r in enumerate(route_list):
            if idx2 != ri and idx2 != rj:
                new_route_list.append(r)
        new_route_list.append(merged)
        route_list = new_route_list
        cust_to_route.clear()
        endpoints.clear()
        for idx3, r in enumerate(route_list):
            interior = r[1:-1]
            for c in interior:
                cust_to_route[c] = idx3
            if interior:
                first = interior[0]
                last = interior[-1]
            else:
                first = last = None
            endpoints.append((first, last, idx3))

    # If still too many routes, merge the two shortest
    while len(route_list) > truck_count:
        dists = [(route_distance(r), idx) for idx, r in enumerate(route_list)]
        dists.sort(key=lambda x: (x[0], x[1]))
        idx1 = dists[0][1]
        idx2 = dists[1][1]
        r1 = route_list[idx1]
        r2 = route_list[idx2]
        merged = r1[:-1] + r2[1:]
        new_route_list = [r for i, r in enumerate(route_list) if i != idx1 and i != idx2]
        new_route_list.append(merged)
        route_list = new_route_list

    report_best_vrp(route_list)

    # Improvement loop
    max_iter = min(300, n * truck_count)
    for iteration in range(max_iter):
        improved = False
        dists = [route_distance(r) for r in route_list]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        interior = route_list[max_idx][1:-1]
        if not interior:
            break

        # Inter-route relocate from longest route
        for cust in interior:
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = route_list[other_idx]
                best_pos = None
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
        if improved:
            continue

        # Inter-route swap between longest and another route
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_interior = route_list[other_idx][1:-1]
            if not other_interior:
                continue
            for cust_max in interior:
                for cust_other in other_interior:
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
        if improved:
            continue

        # Intra-route Or-opt (segment lengths 2 and 3) on the longest route
        route = route_list[max_idx]
        if len(route) > 3:
            best_route = route[:]
            best_dist = route_distance(route)
            for seg_len in [2, 3]:
                if seg_len >= len(route)-1:
                    continue
                for start in range(1, len(route)-seg_len):
                    end = start + seg_len - 1
                    segment = route[start:end+1]
                    remaining = route[:start] + route[end+1:]
                    for pos in range(1, len(remaining)):
                        new_route = remaining[:pos] + segment + remaining[pos:]
                        if new_route[0] != 0 or new_route[-1] != 0:
                            continue
                        d = route_distance(new_route)
                        if d < best_dist:
                            best_dist = d
                            best_route = new_route
            if best_route != route:
                new_routes = [list(r) for r in route_list]
                new_routes[max_idx] = best_route
                new_max = max(route_distance(r) for r in new_routes)
                if new_max < best_max - 1e-12:
                    route_list = new_routes
                    report_best_vrp(route_list)
                    improved = True
        if improved:
            continue

        # Ruin-recreate: remove up to 4 customers farthest from depot from longest route
        worst_route = route_list[max_idx]
        interior = worst_route[1:-1]
        if len(interior) < 1:
            break
        # Compute distances from depot for each customer in interior
        depot_dists = [(distance_matrix[0, c], c) for c in interior]
        depot_dists.sort(key=lambda x: (-x[0], x[1]))
        remove_cnt = min(len(interior), 4)
        to_remove = [depot_dists[k][1] for k in range(remove_cnt)]
        # Remove them from all routes
        new_routes = []
        for r in route_list:
            new_route = [c for c in r if c not in to_remove]
            if new_route[0] != 0:
                new_route = [0] + new_route
            if new_route[-1] != 0:
                new_route.append(0)
            new_routes.append(new_route)
        # Repair using regret-2 (deterministic)
        repair_routes = [r[1:-1] for r in new_routes]
        unassigned = sorted(to_remove)
        while unassigned:
            best_regret = -1e100
            best_cust = None
            best_route_idx = None
            best_pos = None
            for cust in unassigned:
                insertions = []
                for r_idx, route in enumerate(repair_routes):
                    if not route:
                        delta = distance_matrix[0][cust] + distance_matrix[cust][0]
                        insertions.append((delta, r_idx, 0))
                    else:
                        best_delta = float('inf')
                        best_p = 0
                        for pos in range(len(route)+1):
                            if pos == 0:
                                prev = 0
                                nxt = route[0]
                            elif pos == len(route):
                                prev = route[-1]
                                nxt = 0
                            else:
                                prev = route[pos-1]
                                nxt = route[pos]
                            delta = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                            if delta < best_delta:
                                best_delta = delta
                                best_p = pos
                        insertions.append((best_delta, r_idx, best_p))
                insertions.sort(key=lambda x: (x[0], x[2]))
                best = insertions[0][0]
                second = insertions[1][0] if len(insertions) > 1 else best
                regret = second - best
                if regret > best_regret or (regret == best_regret and (best_cust is None or cust < best_cust)):
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = insertions[0][1]
                    best_pos = insertions[0][2]
            if best_cust is None:
                break
            repair_routes[best_route_idx].insert(best_pos, best_cust)
            unassigned.remove(best_cust)
        new_full_routes = [[0] + r + [0] for r in repair_routes]
        new_max = max(route_distance(r) for r in new_full_routes)
        if new_max < best_max - 1e-12:
            route_list = new_full_routes
            report_best_vrp(route_list)
        else:
            break

    final_routes = best_routes if best_routes is not None else route_list
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes