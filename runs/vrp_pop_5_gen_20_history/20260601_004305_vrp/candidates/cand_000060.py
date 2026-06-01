def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_route_dist(routes):
        return max(route_dist(r) for r in routes)

    def greedy_insertion(routes, customer):
        best_inc = float('inf')
        best_ri = -1
        best_pos = -1
        for ri, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_dist = route_dist(route[:pos] + [customer] + route[pos:])
                old_dist = route_dist(route)
                inc = new_dist - old_dist
                if inc < best_inc or (inc == best_inc and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                    best_inc = inc
                    best_ri = ri
                    best_pos = pos
        routes[best_ri].insert(best_pos, customer)
        return routes

    # Deterministic construction: sort customers by distance to depot
    customers = list(range(1, n))
    customers.sort(key=lambda c: distance_matrix[0, c])
    routes = [[0, 0] for _ in range(truck_count)]
    for c in customers:
        routes = greedy_insertion(routes, c)

    best_routes = [list(r) for r in routes]
    best_max = max_route_dist(routes)
    report_best_vrp(best_routes)

    # Relocate improvement
    improved = True
    max_iter = n * truck_count
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for cust in range(1, n):
            ri = None
            pos = None
            for ridx, route in enumerate(routes):
                if cust in route:
                    ri = ridx
                    pos = route.index(cust)
                    break
            if ri is None:
                continue
            old_route = routes[ri][:]
            routes[ri].pop(pos)
            best_ri = -1
            best_pos = -1
            best_new_max = float('inf')
            for other_ri, other_route in enumerate(routes):
                if other_ri == ri:
                    continue
                for p in range(1, len(other_route)):
                    new_other = other_route[:p] + [cust] + other_route[p:]
                    d_ri = route_dist(routes[ri]) if len(routes[ri]) > 1 else 0
                    d_other = route_dist(new_other)
                    max_other = max(d_ri, d_other)
                    for idx, r in enumerate(routes):
                        if idx != ri and idx != other_ri:
                            d = route_dist(r)
                            if d > max_other:
                                max_other = d
                    if max_other < best_new_max:
                        best_new_max = max_other
                        best_ri = other_ri
                        best_pos = p
            if best_new_max < best_max and best_ri != -1:
                routes[best_ri].insert(best_pos, cust)
                improved = True
                cur_max = best_new_max
                if cur_max < best_max:
                    best_max = cur_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
            else:
                routes[ri] = old_route
    return best_routes