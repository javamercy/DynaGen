def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]

    for cust in customers:
        best_max = float('inf')
        best_total = float('inf')
        best_ri = -1
        best_pi = -1
        for ri, route in enumerate(routes):
            for pi in range(1, len(route)):
                new_route = route[:pi] + [cust] + route[pi:]
                new_routes = routes.copy()
                new_routes[ri] = new_route
                max_d = 0
                total_d = 0
                for r in new_routes:
                    d = 0
                    for j in range(len(r) - 1):
                        d += distance_matrix[r[j], r[j+1]]
                    max_d = max(max_d, d)
                    total_d += d
                if max_d < best_max or (max_d == best_max and total_d < best_total):
                    best_max = max_d
                    best_total = total_d
                    best_ri = ri
                    best_pi = pi
        route = routes[best_ri]
        routes[best_ri] = route[:best_pi] + [cust] + route[best_pi:]

    def route_distance(route):
        d = 0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    dists = [route_distance(r) for r in routes]
    best_max = max(dists)
    best_routes = [r[:] for r in routes]
    report_best_vrp(best_routes)

    max_iter = len(customers) * 10
    for _ in range(max_iter):
        improved = False

        # relocate
        for ri in range(truck_count):
            route = routes[ri]
            for i in range(1, len(route) - 1):
                cust = route[i]
                new_route_ri = route[:i] + route[i+1:]
                for rj in range(truck_count):
                    other_route = new_route_ri if rj == ri else routes[rj]
                    for j in range(1, len(other_route)):
                        new_other = other_route[:j] + [cust] + other_route[j:]
                        new_routes = routes.copy()
                        if rj == ri:
                            new_routes[ri] = new_other
                        else:
                            new_routes[ri] = new_route_ri
                            new_routes[rj] = new_other
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max - 1e-9:
                            routes = new_routes
                            dists = [route_distance(r) for r in routes]
                            best_max = max(dists)
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
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

        # swap
        for ri in range(truck_count):
            route1 = routes[ri]
            for i in range(1, len(route1) - 1):
                for rj in range(truck_count):
                    route2 = routes[rj]
                    for j in range(1, len(route2) - 1):
                        if ri == rj and i >= j:
                            continue
                        new_route1 = route1.copy()
                        new_route2 = route2.copy()
                        new_route1[i] = route2[j]
                        new_route2[j] = route1[i]
                        new_routes = routes.copy()
                        new_routes[ri] = new_route1
                        new_routes[rj] = new_route2
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max - 1e-9:
                            routes = new_routes
                            dists = [route_distance(r) for r in routes]
                            best_max = max(dists)
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
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

        # 2-opt within route
        for ri in range(truck_count):
            route = routes[ri]
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_d = route_distance(new_route)
                    new_dists = dists.copy()
                    new_dists[ri] = new_d
                    new_max = max(new_dists)
                    if new_max < best_max - 1e-9:
                        routes[ri] = new_route
                        dists[ri] = new_d
                        best_max = new_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            break

    return best_routes