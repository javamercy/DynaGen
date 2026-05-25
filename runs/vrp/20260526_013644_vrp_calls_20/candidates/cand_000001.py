import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # initial routes: each customer alone
    routes = [[0, c, 0] for c in customers]
    # compute savings and sort
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))
    # helper functions
    def route_distance(route):
        total = 0.0
        for a, b in zip(route, route[1:]):
            total += distance_matrix[a][b]
        return total
    def is_endpoint(route, c):
        return route[1] == c or route[-2] == c
    # merge until we have exactly truck_count routes
    active_routes = list(routes)
    # mapping customer to route index (for speed)
    def build_cust_to_route():
        d = {}
        for idx, r in enumerate(active_routes):
            for c in r[1:-1]:
                d[c] = idx
        return d
    cust_to_route = build_cust_to_route()
    # while too many routes, try to merge using savings order
    while len(active_routes) > truck_count:
        best_merge = None  # (i, j) pair to merge
        best_i_idx = None
        best_j_idx = None
        best_saving = -1e9
        # scan all pairs of routes
        for idx_i in range(len(active_routes)):
            ri = active_routes[idx_i]
            if len(ri) <= 2:
                continue
            for idx_j in range(idx_i + 1, len(active_routes)):
                rj = active_routes[idx_j]
                if len(rj) <= 2:
                    continue
                # endpoints of ri
                ri_endpoints = set()
                if ri[1] != 0:
                    ri_endpoints.add(ri[1])
                if ri[-2] != 0:
                    ri_endpoints.add(ri[-2])
                rj_endpoints = set()
                if rj[1] != 0:
                    rj_endpoints.add(rj[1])
                if rj[-2] != 0:
                    rj_endpoints.add(rj[-2])
                for i in ri_endpoints:
                    for j in rj_endpoints:
                        if i == j:
                            continue
                        # check if i and j are at opposite ends
                        if (ri[-2] == i and rj[1] == j) or (ri[1] == i and rj[-2] == j):
                            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                            if s > best_saving:
                                best_saving = s
                                best_merge = (i, j)
                                best_i_idx = idx_i
                                best_j_idx = idx_j
        if best_merge is None:
            # fallback: merge any two routes arbitrarily
            if len(active_routes) >= 2:
                ri = active_routes[0]
                rj = active_routes[1]
                new_route = ri[:-1] + rj[1:]
                active_routes[0] = new_route
                active_routes.pop(1)
            else:
                break
        else:
            i, j = best_merge
            ri = active_routes[best_i_idx]
            rj = active_routes[best_j_idx]
            if ri[-2] == i and rj[1] == j:
                new_route = ri[:-1] + rj[1:]
            else:
                new_route = rj[:-1] + ri[1:]
            active_routes[best_i_idx] = new_route
            active_routes.pop(best_j_idx)
        cust_to_route = build_cust_to_route()
    # add empty routes if needed
    while len(active_routes) < truck_count:
        active_routes.append([0, 0])
    # compute initial best
    best_routes = [list(r) for r in active_routes]
    best_max = max(route_distance(r) for r in best_routes)
    # report initial solution
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass
    # local search: move customers from longest route to others to reduce max
    for _ in range(10):
        current_max = max(route_distance(r) for r in active_routes)
        max_idx = max(range(len(active_routes)), key=lambda idx: route_distance(active_routes[idx]))
        max_route = active_routes[max_idx]
        if len(max_route) <= 2:
            break
        best_new_max = current_max
        best_new_routes = None
        # iterate over each customer in max_route
        for cust in max_route[1:-1]:
            for other_idx in range(len(active_routes)):
                if other_idx == max_idx:
                    continue
                other_route = active_routes[other_idx]
                # try inserting at each position in other_route
                for pos in range(1, len(other_route)):  # insert after node at index pos-1
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    # remove cust from max_route
                    new_max_route = [x for x in max_route if x != cust]
                    new_routes = [list(active_routes[i]) for i in range(len(active_routes))]
                    new_routes[max_idx] = new_max_route
                    new_routes[other_idx] = new_other
                    new_max = max(route_distance(r) for r in new_routes)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_new_routes = new_routes
        if best_new_routes is not None and best_new_max < current_max:
            active_routes = best_new_routes
            best_routes = [list(r) for r in active_routes]
            try:
                report_best_vrp(best_routes)
            except NameError:
                pass
        else:
            break
    return best_routes