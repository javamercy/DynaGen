import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        max_iter = len(route) * 5
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d_new = route_distance(new_route)
                    if d_new < route_distance(route) - 1e-12:
                        route = new_route
                        improved = True
            iter_count += 1
        return route

    # Construction: sort customers by distance from depot
    depot_dists = [distance_matrix[0, i] for i in customers]
    sorted_cust = sorted(customers, key=lambda i: depot_dists[i-1])

    routes = [[0, 0] for _ in range(truck_count)]
    for cust in sorted_cust:
        best_truck = 0
        best_increase = float('inf')
        for t in range(truck_count):
            if len(routes[t]) == 2:
                increase = distance_matrix[0, cust] + distance_matrix[cust, 0]
            else:
                last = routes[t][-2]
                increase = distance_matrix[last, cust] + distance_matrix[cust, 0] - distance_matrix[last, 0]
            if increase < best_increase - 1e-12:
                best_increase = increase
                best_truck = t
        routes[best_truck].insert(-1, cust)

    # Apply 2-opt to all routes
    for t in range(truck_count):
        routes[t] = two_opt(routes[t])

    best_routes = [r[:] for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    # Inter-route improvement: move customer from longest route to shortest if it reduces max
    max_iter = n * truck_count
    iter_count = 0
    while iter_count < max_iter:
        current_max = max(route_distance(r) for r in routes)
        max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
        min_idx = min(range(truck_count), key=lambda t: route_distance(routes[t]))
        if max_idx == min_idx:
            break
        max_route = routes[max_idx]
        min_route = routes[min_idx]
        best_move = None
        best_new_max = current_max
        for i in range(1, len(max_route)-1):
            cust = max_route[i]
            new_max = max_route[:i] + max_route[i+1:]
            for j in range(1, len(min_route)):
                new_min = min_route[:j] + [cust] + min_route[j:]
                other_max = 0.0
                for t in range(truck_count):
                    if t != max_idx and t != min_idx:
                        d = route_distance(routes[t])
                        if d > other_max:
                            other_max = d
                cand_max = max(route_distance(new_max), route_distance(new_min), other_max)
                if cand_max < best_new_max - 1e-12:
                    best_new_max = cand_max
                    best_move = (i, j, new_max, new_min)
        if best_move is None:
            break
        i, j, new_max, new_min = best_move
        routes[max_idx] = new_max
        routes[min_idx] = new_min
        routes[max_idx] = two_opt(routes[max_idx])
        routes[min_idx] = two_opt(routes[min_idx])
        iter_count += 1
        current_max = max(route_distance(r) for r in routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    # Final 2-opt on best routes
    for t in range(truck_count):
        best_routes[t] = two_opt(best_routes[t])
    report_best_vrp(best_routes)
    return best_routes