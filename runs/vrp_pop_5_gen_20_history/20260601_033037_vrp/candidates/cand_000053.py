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

    # ---- Clarke-Wright savings construction ----
    route_list = [[0, c, 0] for c in customers]
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append(( -s, i, j ))
    heapq.heapify(savings)

    cust_to_route = {}
    route_endpoints = []
    for idx, route in enumerate(route_list):
        if len(route) == 3:
            cust = route[1]
            cust_to_route[cust] = idx
            route_endpoints.append((cust, cust, idx))

    while len(route_list) > truck_count and savings:
        neg_s, i, j = heapq.heappop(savings)
        if i not in cust_to_route or j not in cust_to_route:
            continue
        ri = cust_to_route[i]
        rj = cust_to_route[j]
        if ri == rj:
            continue
        first_i, last_i, _ = route_endpoints[ri]
        first_j, last_j, _ = route_endpoints[rj]
        if i == last_i and j == first_j:
            route_i = route_list[ri]
            route_j = route_list[rj]
            new_route = route_i[:-1] + route_j[1:]
        elif i == first_i and j == last_j:
            route_i = route_list[ri]
            route_j = route_list[rj]
            new_route = route_j[:-1] + route_i[1:]
        elif j == last_j and i == first_i:
            route_i = route_list[ri]
            route_j = route_list[rj]
            new_route = route_j[:-1] + route_i[1:]
        elif j == first_j and i == last_i:
            route_i = route_list[ri]
            route_j = route_list[rj]
            new_route = route_i[:-1] + route_j[1:]
        else:
            continue
        new_route_list = []
        for idx, r in enumerate(route_list):
            if idx == ri or idx == rj:
                continue
            new_route_list.append(r)
        new_route_list.append(new_route)
        route_list = new_route_list
        cust_to_route.clear()
        route_endpoints.clear()
        for idx2, r in enumerate(route_list):
            interior = r[1:-1]
            for c in interior:
                cust_to_route[c] = idx2
            first_cust = interior[0] if interior else None
            last_cust = interior[-1] if interior else None
            if first_cust is None:
                route_endpoints.append((None, None, idx2))
            else:
                route_endpoints.append((first_cust, last_cust, idx2))

    while len(route_list) > truck_count:
        dists = [(route_distance(r), idx) for idx, r in enumerate(route_list)]
        dists.sort(key=lambda x: x[0])
        idx1 = dists[0][1]
        idx2 = dists[1][1]
        r1 = route_list[idx1]
        r2 = route_list[idx2]
        new_route = r1[:-1] + r2[1:]
        new_route_list = []
        for i, r in enumerate(route_list):
            if i == idx1 or i == idx2:
                continue
            new_route_list.append(r)
        new_route_list.append(new_route)
        route_list = new_route_list

    report_best_vrp(route_list)

    # ---- Steepest improvement ----
    max_iter = len(customers) * truck_count * 2
    for _ in range(max_iter):
        improved = False
        # Evaluate all relocate moves
        best_relocate = None
        best_relocate_new_max = float('inf')
        for max_idx in range(truck_count):
            dists = [route_distance(r) for r in route_list]
            # We'll always try to reduce the current max, but for steepest we consider all max_idx
            pass
        # Actually, we want to reduce the max distance, so we consider moves that affect the max route
        # But steepest descent over all routes: we consider any relocate that reduces max distance
        # Simpler: iterate over all relocate moves (customer from any route to any other), compute new max, keep best
        for src_idx in range(truck_count):
            src_route = route_list[src_idx]
            interior = src_route[1:-1]
            for cust in interior:
                for dst_idx in range(truck_count):
                    if dst_idx == src_idx:
                        continue
                    dst_route = route_list[dst_idx]
                    best_pos = None
                    best_delta = float('inf')
                    for pos in range(1, len(dst_route)):
                        prev = dst_route[pos-1]
                        nxt = dst_route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = pos
                    # Temporarily construct new routes to compute new max
                    new_routes = [list(r) for r in route_list]
                    new_routes[src_idx].remove(cust)
                    new_routes[dst_idx].insert(best_pos, cust)
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_relocate_new_max:
                        best_relocate_new_max = new_max
                        best_relocate = (src_idx, dst_idx, cust, best_pos, new_routes)
        # Evaluate all swap moves
        best_swap = None
        best_swap_new_max = float('inf')
        for idx1 in range(truck_count):
            route1 = route_list[idx1]
            interior1 = route1[1:-1]
            for cust1 in interior1:
                for idx2 in range(idx1+1, truck_count):
                    route2 = route_list[idx2]
                    interior2 = route2[1:-1]
                    for cust2 in interior2:
                        new_routes = [list(r) for r in route_list]
                        i1 = new_routes[idx1].index(cust1)
                        i2 = new_routes[idx2].index(cust2)
                        new_routes[idx1][i1] = cust2
                        new_routes[idx2][i2] = cust1
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_swap_new_max:
                            best_swap_new_max = new_max
                            best_swap = (new_routes, new_max)
        # Apply the best move among relocate and swap
        if best_relocate is not None and best_relocate_new_max < best_max - 1e-12:
            route_list = best_relocate[4]
            report_best_vrp(route_list)
            improved = True
            continue
        if best_swap is not None and best_swap_new_max < best_max - 1e-12:
            route_list = best_swap[0]
            report_best_vrp(route_list)
            improved = True
            continue
        # 2-opt on each route (best improvement)
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