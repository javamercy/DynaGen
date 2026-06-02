import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # Initial routes
    routes = [[0, 0] for _ in range(truck_count)]
    unvisited = list(range(1, n))
    # Cheapest insertion
    while unvisited:
        best_increase = np.inf
        best_truck = -1
        best_pos = -1
        best_customer = -1
        for cust in unvisited:
            for t in range(truck_count):
                route = routes[t]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    next_ = route[pos]
                    increase = distance_matrix[prev][cust] + distance_matrix[cust][next_] - distance_matrix[prev][next_]
                    if increase < best_increase or (increase == best_increase and cust < best_customer):
                        best_increase = increase
                        best_truck = t
                        best_pos = pos
                        best_customer = cust
        if best_customer == -1:
            break
        routes[best_truck].insert(best_pos, best_customer)
        unvisited.remove(best_customer)
    # Helper: compute route distance
    def route_distance(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    # Helper: compute max distance
    def max_distance(routes):
        return max(route_distance(route) for route in routes)
    best_max = max_distance(routes)
    report_best_vrp(routes)
    # Local search
    max_iter = n * 2
    for _ in range(max_iter):
        improved = False
        # Relocate (cross-route)
        for t1 in range(truck_count):
            for t2 in range(truck_count):
                if t1 == t2:
                    continue
                route1 = routes[t1]
                route2 = routes[t2]
                for i in range(1, len(route1)-1):
                    cust = route1[i]
                    new_route1 = route1[:i] + route1[i+1:]
                    for pos in range(1, len(route2)):
                        new_route2 = route2[:pos] + [cust] + route2[pos:]
                        # Temporarily update routes to compute distances
                        old_route1 = routes[t1]
                        old_route2 = routes[t2]
                        routes[t1] = new_route1
                        routes[t2] = new_route2
                        new_max = max_distance(routes)
                        if new_max < best_max:
                            best_max = new_max
                            improved = True
                            break
                        else:
                            routes[t1] = old_route1
                            routes[t2] = old_route2
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            report_best_vrp(routes)
            continue
        # Swap (cross-route)
        for t1 in range(truck_count):
            for t2 in range(t1+1, truck_count):
                route1 = routes[t1]
                route2 = routes[t2]
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        cust1 = route1[i]
                        cust2 = route2[j]
                        new_route1 = route1[:i] + [cust2] + route1[i+1:]
                        new_route2 = route2[:j] + [cust1] + route2[j+1:]
                        old_route1 = routes[t1]
                        old_route2 = routes[t2]
                        routes[t1] = new_route1
                        routes[t2] = new_route2
                        new_max = max_distance(routes)
                        if new_max < best_max:
                            best_max = new_max
                            improved = True
                            break
                        else:
                            routes[t1] = old_route1
                            routes[t2] = old_route2
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            report_best_vrp(routes)
            continue
        # Intra-route 2-opt
        for t in range(truck_count):
            route = routes[t]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-2):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    old_route = routes[t]
                    routes[t] = new_route
                    new_max = max_distance(routes)
                    if new_max < best_max:
                        best_max = new_max
                        improved = True
                        break
                    else:
                        routes[t] = old_route
                if improved:
                    break
            if improved:
                break
        if improved:
            report_best_vrp(routes)
            continue
        # Cross-route 2-opt*
        for t1 in range(truck_count):
            for t2 in range(t1+1, truck_count):
                route1 = routes[t1]
                route2 = routes[t2]
                for i in range(1, len(route1)-2):
                    for j in range(1, len(route2)-2):
                        # new route1 = route1[:i+1] + route2[j+1:]
                        # new route2 = route2[:j+1] + route1[i+1:]
                        new_route1 = route1[:i+1] + route2[j+1:]
                        new_route2 = route2[:j+1] + route1[i+1:]
                        # Check that both end with 0 (should always be true)
                        old_route1 = routes[t1]
                        old_route2 = routes[t2]
                        routes[t1] = new_route1
                        routes[t2] = new_route2
                        new_max = max_distance(routes)
                        if new_max < best_max:
                            best_max = new_max
                            improved = True
                            break
                        else:
                            routes[t1] = old_route1
                            routes[t2] = old_route2
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            report_best_vrp(routes)
        else:
            break
    return routes