import numpy as np
import heapq

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        new_max = max(route_distance(r) for r in routes)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [list(r) for r in routes]

    # Step 1: Clarke-Wright savings construction
    routes = [[0, c, 0] for c in customers]
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                heapq.heappush(savings, (-s, i, j))
    # Track endpoints for merging
    first = {}
    last = {}
    route_of_customer = {}
    for idx, r in enumerate(routes):
        c = r[1]
        first[c] = c
        last[c] = c
        route_of_customer[c] = idx

    while len(routes) > truck_count and savings:
        neg_s, i, j = heapq.heappop(savings)
        if i not in route_of_customer or j not in route_of_customer:
            continue
        ri = route_of_customer[i]
        rj = route_of_customer[j]
        if ri == rj:
            continue
        if last.get(i) == i and first.get(j) == j:
            # Merge ri and rj: attach j after i
            route_i = routes[ri]
            route_j = routes[rj]
            merged = route_i[:-1] + route_j[1:]
            new_first = first[route_i[1]] if len(route_i) > 2 else None
            new_last = last[route_j[-2]] if len(route_j) > 2 else None
            # Update routes
            new_routes = [r for idx_r, r in enumerate(routes) if idx_r not in (ri, rj)]
            new_routes.append(merged)
            routes = new_routes
            # Update tracking
            route_of_customer.clear()
            first.clear()
            last.clear()
            for idx_r, r in enumerate(routes):
                interior = r[1:-1]
                if interior:
                    for c in interior:
                        route_of_customer[c] = idx_r
                    first[interior[0]] = interior[0]
                    last[interior[-1]] = interior[-1]
        elif last.get(j) == j and first.get(i) == i:
            # Merge rj and ri: attach i after j
            route_i = routes[ri]
            route_j = routes[rj]
            merged = route_j[:-1] + route_i[1:]
            new_first = first[route_j[1]] if len(route_j) > 2 else None
            new_last = last[route_i[-2]] if len(route_i) > 2 else None
            new_routes = [r for idx_r, r in enumerate(routes) if idx_r not in (ri, rj)]
            new_routes.append(merged)
            routes = new_routes
            route_of_customer.clear()
            first.clear()
            last.clear()
            for idx_r, r in enumerate(routes):
                interior = r[1:-1]
                if interior:
                    for c in interior:
                        route_of_customer[c] = idx_r
                    first[interior[0]] = interior[0]
                    last[interior[-1]] = interior[-1]
        # other endpoint combinations are not standard savings merge, skip

    # If still too many routes, merge the two shortest
    while len(routes) > truck_count:
        dists = [(route_distance(r), idx) for idx, r in enumerate(routes)]
        dists.sort(key=lambda x: (x[0], x[1]))
        idx1 = dists[0][1]
        idx2 = dists[1][1]
        r1 = routes[idx1]
        r2 = routes[idx2]
        merged = r1[:-1] + r2[1:]
        new_routes = [r for i, r in enumerate(routes) if i not in (idx1, idx2)]
        new_routes.append(merged)
        routes = new_routes

    report_best_vrp(routes)

    # Step 2: Improvement loop (relocate, swap, 2-opt)
    max_iter = min(200, n * 2)
    for _ in range(max_iter):
        improved = False
        # Determine longest route
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        if len(routes[max_idx]) <= 3:
            break
        # Relocate: move customer from longest route to another
        interior = routes[max_idx][1:-1]
        for cust in interior:
            best_delta = float('inf')
            best_route_idx = None
            best_pos = None
            for other_idx in range(len(routes)):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    prev = other_route[pos-1]
                    nxt = other_route[pos]
                    delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if delta < best_delta:
                        best_delta = delta
                        best_route_idx = other_idx
                        best_pos = pos
            if best_route_idx is not None:
                new_routes = [list(r) for r in routes]
                new_routes[max_idx].remove(cust)
                new_routes[best_route_idx].insert(best_pos, cust)
                new_max = max(route_distance(r) for r in new_routes)
                if new_max < best_max - 1e-12:
                    routes = new_routes
                    report_best_vrp(routes)
                    improved = True
                    break
        if improved:
            continue
        # Swap: between longest and another
        for other_idx in range(len(routes)):
            if other_idx == max_idx:
                continue
            other_interior = routes[other_idx][1:-1]
            if not other_interior:
                continue
            for cust_max in interior:
                for cust_other in other_interior:
                    new_routes = [list(r) for r in routes]
                    idx_max = new_routes[max_idx].index(cust_max)
                    idx_other = new_routes[other_idx].index(cust_other)
                    new_routes[max_idx][idx_max] = cust_other
                    new_routes[other_idx][idx_other] = cust_max
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_max - 1e-12:
                        routes = new_routes
                        report_best_vrp(routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 2-opt on each route (try to reduce its own distance, may indirectly reduce max)
        for idx in range(len(routes)):
            route = routes[idx]
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_dist = route_distance(route)
            improved_2opt = False
            for a in range(1, len(route)-2):
                for b in range(a+1, len(route)-1):
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist - 1e-12:
                        best_dist = new_dist
                        best_route = new_route
                        improved_2opt = True
                        break
                if improved_2opt:
                    break
            if improved_2opt:
                routes[idx] = best_route
                new_max = max(route_distance(r) for r in routes)
                if new_max < best_max - 1e-12:
                    report_best_vrp(routes)
                improved = True
                break
        if not improved:
            break

    final_routes = best_routes if best_routes is not None else routes
    while len(final_routes) < truck_count:
        final_routes.append([0, 0])
    return final_routes