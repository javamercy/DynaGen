import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Initialize routes as list of customer lists (no depot)
    routes = [[] for _ in range(truck_count)]
    unassigned = sorted(customers)

    def route_distance(route):
        if not route:
            return 0.0
        d = distance_matrix[0, route[0]] + distance_matrix[route[-1], 0]
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_max():
        return max(route_distance(r) for r in routes)

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(full_routes):
        nonlocal best_routes, best_max
        maxd = max(sum(distance_matrix[full_routes[i][j]][full_routes[i][j+1]] for j in range(len(full_routes[i])-1)) for i in range(len(full_routes)))
        if maxd < best_max:
            best_max = maxd
            best_routes = [list(r) for r in full_routes]

    def full_routes_from_routes(rts):
        return [[0] + r + [0] for r in rts]

    # Regret insertion construction
    while unassigned:
        best_regret = -1e100
        best_cust = None
        best_route_idx = None
        best_pos = None
        for cust in sorted(unassigned):
            route_insertions = []
            for r_idx, route in enumerate(routes):
                if not route:
                    delta = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    route_insertions.append((delta, r_idx, 0))
                else:
                    min_delta = float('inf')
                    best_p = None
                    for pos in range(len(route)+1):
                        if pos == 0:
                            prev = 0
                            next = route[0]
                        elif pos == len(route):
                            prev = route[-1]
                            next = 0
                        else:
                            prev = route[pos-1]
                            next = route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, next] - distance_matrix[prev, next]
                        if delta < min_delta:
                            min_delta = delta
                            best_p = pos
                    route_insertions.append((min_delta, r_idx, best_p))
            route_insertions.sort(key=lambda x: x[0])
            best = route_insertions[0][0]
            second = route_insertions[1][0] if len(route_insertions) > 1 else best
            regret = second - best
            if regret > best_regret:
                best_regret = regret
                best_cust = cust
                best_route_idx = route_insertions[0][1]
                best_pos = route_insertions[0][2]
        # Insert best customer
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    report_best_vrp(full_routes_from_routes(routes))

    # Improvement: relocate and 2-opt
    n_customers = len(customers)
    max_iters = n_customers * truck_count
    for _ in range(max_iters):
        improved = False
        # Find route with maximum distance
        dists = [route_distance(r) for r in routes]
        max_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
        route_long = routes[max_idx]
        if not route_long:
            break
        # Try moving each customer from longest route to another
        for cust in list(route_long):
            pos = route_long.index(cust)
            # Removal delta from longest
            if len(route_long) == 1:
                delta_rem = 0
            else:
                if pos == 0:
                    prev = 0
                    next = route_long[1]
                elif pos == len(route_long)-1:
                    prev = route_long[-2]
                    next = 0
                else:
                    prev = route_long[pos-1]
                    next = route_long[pos+1]
                delta_rem = distance_matrix[prev, next] - distance_matrix[prev, cust] - distance_matrix[cust, next]
            new_long = route_long[:pos] + route_long[pos+1:]
            # Try inserting into other routes
            for r_idx in range(truck_count):
                if r_idx == max_idx:
                    continue
                other = routes[r_idx]
                # Find best insertion position
                if not other:
                    delta_ins = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    best_pos_other = 0
                else:
                    best_delta = float('inf')
                    best_pos_other = None
                    for p in range(len(other)+1):
                        if p == 0:
                            prev = 0
                            nxt = other[0]
                        elif p == len(other):
                            prev = other[-1]
                            nxt = 0
                        else:
                            prev = other[p-1]
                            nxt = other[p]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        if delta < best_delta:
                            best_delta = delta
                            best_pos_other = p
                    delta_ins = best_delta
                # Evaluate new max
                new_other = other[:best_pos_other] + [cust] + other[best_pos_other:]
                new_routes = routes[:]
                new_routes[max_idx] = new_long
                new_routes[r_idx] = new_other
                new_max = max(route_distance(r) for r in new_routes)
                if new_max < best_max:
                    routes = new_routes
                    report_best_vrp(full_routes_from_routes(routes))
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # Try swap between two routes
        for i in range(truck_count):
            for j in range(i+1, truck_count):
                route_i = routes[i]
                route_j = routes[j]
                if not route_i or not route_j:
                    continue
                for cust_i in route_i:
                    for cust_j in route_j:
                        pos_i = route_i.index(cust_i)
                        pos_j = route_j.index(cust_j)
                        new_i = route_i[:pos_i] + [cust_j] + route_i[pos_i+1:]
                        new_j = route_j[:pos_j] + [cust_i] + route_j[pos_j+1:]
                        new_routes = routes[:]
                        new_routes[i] = new_i
                        new_routes[j] = new_j
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max:
                            routes = new_routes
                            report_best_vrp(full_routes_from_routes(routes))
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 2-opt on each route
        for idx in range(truck_count):
            route = routes[idx]
            if len(route) < 3:
                continue
            for _ in range(len(route)):
                best_route = route[:]
                best_dist = route_distance(route)
                found = False
                for a in range(len(route)-1):
                    for b in range(a+2, len(route)+1):
                        if b - a < 2:
                            continue
                        new_route = route[:a] + route[a:b][::-1] + route[b:]
                        new_dist = route_distance(new_route)
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_route = new_route
                            found = True
                            break
                    if found:
                        break
                if found:
                    routes[idx] = best_route
                    report_best_vrp(full_routes_from_routes(routes))
                    improved = True
                    break
                else:
                    break
        if not improved:
            break

    # Ensure exactly truck_count routes
    while len(best_routes) > truck_count:
        # Merge two shortest routes
        dists = []
        for idx, r in enumerate(best_routes):
            d = sum(distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1))
            dists.append((d, idx))
        dists.sort(key=lambda x: x[0])
        i = dists[0][1]
        j = dists[1][1]
        new_route = best_routes[i][:-1] + best_routes[j][1:]
        best_routes[i] = new_route
        best_routes.pop(j)
    while len(best_routes) < truck_count:
        best_routes.append([0,0])
    return best_routes