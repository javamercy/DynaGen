import numpy as np
import heapq

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

    # Clarke-Wright savings construction
    route_list = [[0, c, 0] for c in customers]
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append((-s, i, j))
    heapq.heapify(savings)

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

    while len(route_list) > truck_count and savings:
        _, i, j = heapq.heappop(savings)
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
        new_route_list = [r for idx, r in enumerate(route_list) if idx != ri and idx != rj]
        new_route_list.append(merged)
        route_list = new_route_list
        cust_to_route.clear()
        endpoints.clear()
        for idx, r in enumerate(route_list):
            interior = r[1:-1]
            for c in interior:
                cust_to_route[c] = idx
            if interior:
                first = interior[0]
                last = interior[-1]
            else:
                first = last = None
            endpoints.append((first, last, idx))

    while len(route_list) > truck_count:
        dists = [(route_distance(r), idx) for idx, r in enumerate(route_list)]
        dists.sort(key=lambda x: (x[0], x[1]))
        idx1 = dists[0][1]
        idx2 = dists[1][1]
        r1 = route_list[idx1]
        r2 = route_list[idx2]
        merged = r1[:-1] + r2[1:]
        new_route_list = [r for idx, r in enumerate(route_list) if idx != idx1 and idx != idx2]
        new_route_list.append(merged)
        route_list = new_route_list

    report_best_vrp(route_list)

    max_iter = min(300, n * truck_count)
    for _ in range(max_iter):
        improved = False
        dists = [route_distance(r) for r in route_list]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        interior = route_list[max_idx][1:-1]
        # Inter-route relocate from longest route
        if interior:
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
        if interior:
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
        # Intra-route 2-opt
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
            break

    final_routes = best_routes if best_routes is not None else route_list
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes