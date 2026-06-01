import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    num_customers = len(customers)
    if truck_count >= num_customers:
        routes = [[0, c, 0] for c in customers]
        for _ in range(truck_count - num_customers):
            routes.append([0, 0])
        return routes

    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total

    best_routes = None
    best_max = float('inf')

    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        maxd = max(route_distance(r) for r in routes)
        if maxd < best_max:
            best_max = maxd
            best_routes = [list(r) for r in routes]

    # ---- Regret-2 construction ----
    routes = [[] for _ in range(truck_count)]
    unassigned = sorted(customers)
    while unassigned:
        best_regret = -1e100
        best_cust = None
        best_route_idx = None
        best_pos = None
        for cust in sorted(unassigned):
            insertions = []
            for r_idx, route in enumerate(routes):
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
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    full_routes = [[0] + r + [0] for r in routes]
    report_best_vrp(full_routes)
    current_routes = full_routes
    current_max = best_max

    # ---- Local search helpers ----
    def local_search(routes, best_max):
        improved = True
        while improved:
            improved = False
            # Inter-route relocate
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for cust in sorted(route_i[1:-1]):
                    for j in range(truck_count):
                        if i == j:
                            continue
                        route_j = routes[j]
                        for pos in range(1, len(route_j)):
                            new_routes = [list(r) for r in routes]
                            new_routes[i].remove(cust)
                            new_routes[j].insert(pos, cust)
                            new_max = max(route_distance(r) for r in new_routes)
                            if new_max < best_max:
                                routes = new_routes
                                best_max = new_max
                                report_best_vrp(routes)
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
            # Inter-route swap
            for i in range(truck_count):
                route_i = routes[i]
                if len(route_i) <= 2:
                    continue
                for cust_i in sorted(route_i[1:-1]):
                    for j in range(i+1, truck_count):
                        route_j = routes[j]
                        if len(route_j) <= 2:
                            continue
                        for cust_j in sorted(route_j[1:-1]):
                            new_routes = [list(r) for r in routes]
                            idx_i = new_routes[i].index(cust_i)
                            idx_j = new_routes[j].index(cust_j)
                            new_routes[i][idx_i], new_routes[j][idx_j] = cust_j, cust_i
                            new_max = max(route_distance(r) for r in new_routes)
                            if new_max < best_max:
                                routes = new_routes
                                best_max = new_max
                                report_best_vrp(routes)
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
            # Intra-route 2-opt
            for i in range(truck_count):
                route = routes[i]
                if len(route) <= 3:
                    continue
                best_route = route[:]
                best_dist = route_distance(route)
                found = False
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_route = new_route
                            found = True
                            break
                    if found:
                        break
                if found:
                    routes[i] = best_route
                    new_max = max(route_distance(r) for r in routes)
                    if new_max < best_max:
                        best_max = new_max
                        report_best_vrp(routes)
                    improved = True
                    break
        return routes, best_max

    # ---- Main loop ----
    max_iter = num_customers * truck_count
    for iteration in range(max_iter):
        current_routes, current_max = local_search(current_routes, current_max)
        if current_max == best_max:
            dists = [route_distance(r) for r in current_routes]
            worst_idx = max(range(len(dists)), key=lambda i: (dists[i], i))
            worst_route = current_routes[worst_idx]
            if len(worst_route) <= 3:
                break
            interior = worst_route[1:-1]
            # Adaptive removal: remove up to 20% of interior customers, at least 1
            remove_cnt = max(1, int(len(interior) * 0.2))
            to_remove = sorted(interior)[:remove_cnt]
            new_routes = []
            for r in current_routes:
                new_route = [c for c in r if c not in to_remove]
                if new_route[0] != 0:
                    new_route = [0] + new_route
                if new_route[-1] != 0:
                    new_route.append(0)
                new_routes.append(new_route)
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
            if new_max < best_max:
                current_routes = new_full_routes
                current_max = new_max
                report_best_vrp(current_routes)
            else:
                break
        else:
            continue
    current_routes, current_max = local_search(current_routes, current_max)
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes