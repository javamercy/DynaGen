import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # Initial routes (same as parent)
    routes = [[0, 0] for _ in range(truck_count)]
    unvisited = list(range(1, n))
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

    def route_distance(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

    def max_distance(routes):
        return max(route_distance(route) for route in routes)

    best_max = max_distance(routes)
    report_best_vrp(routes)

    # Define neighborhoods as functions that modify routes in-place and return True if improved
    def relocate(routes):
        nonlocal best_max
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
                        old_route1 = routes[t1]
                        old_route2 = routes[t2]
                        routes[t1] = new_route1
                        routes[t2] = new_route2
                        new_max = max_distance(routes)
                        if new_max < best_max:
                            best_max = new_max
                            return True
                        else:
                            routes[t1] = old_route1
                            routes[t2] = old_route2
        return False

    def swap(routes):
        nonlocal best_max
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
                            return True
                        else:
                            routes[t1] = old_route1
                            routes[t2] = old_route2
        return False

    def intra_2opt(routes):
        nonlocal best_max
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
                        return True
                    else:
                        routes[t] = old_route
        return False

    def cross_2opt_star(routes):
        nonlocal best_max
        for t1 in range(truck_count):
            for t2 in range(t1+1, truck_count):
                route1 = routes[t1]
                route2 = routes[t2]
                for i in range(1, len(route1)-2):
                    for j in range(1, len(route2)-2):
                        new_route1 = route1[:i+1] + route2[j+1:]
                        new_route2 = route2[:j+1] + route1[i+1:]
                        old_route1 = routes[t1]
                        old_route2 = routes[t2]
                        routes[t1] = new_route1
                        routes[t2] = new_route2
                        new_max = max_distance(routes)
                        if new_max < best_max:
                            best_max = new_max
                            return True
                        else:
                            routes[t1] = old_route1
                            routes[t2] = old_route2
        return False

    neighborhoods = [relocate, swap, intra_2opt, cross_2opt_star]
    no_improve_limit = max(1, int(n * 0.5))
    no_improve_count = 0

    while True:
        improved = False
        for i in range(len(neighborhoods)):
            if neighborhoods[i](routes):
                improved = True
                report_best_vrp(routes)
                # Move successful neighborhood to front
                neighborhoods.insert(0, neighborhoods.pop(i))
                break
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= no_improve_limit:
                break

    return routes