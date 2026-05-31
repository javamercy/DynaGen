import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    depot = 0
    routes = [[depot, depot] for _ in range(truck_count)]
    unvisited = list(range(1, n))
    # sort customers by distance from depot descending
    unvisited.sort(key=lambda c: distance_matrix[depot, c], reverse=True)
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def compute_max_distance():
        maxd = 0.0
        for r in routes:
            rd = route_distance(r)
            if rd > maxd:
                maxd = rd
        return maxd
    
    # --- Construction ---
    for cust in unvisited:
        best_max = float('inf')
        best_route = -1
        best_pos = -1
        for t in range(truck_count):
            route = routes[t]
            cur_dist = route_distance(route)
            for pos in range(1, len(route)):
                # compute new route distance for this truck after insertion
                prev = route[pos-1]
                next_ = route[pos]
                inc = distance_matrix[prev][cust] + distance_matrix[cust][next_] - distance_matrix[prev][next_]
                new_dist = cur_dist + inc
                # compute new max considering all other routes unchanged
                current_max = max(new_dist, max(route_distance(routes[tt]) for tt in range(truck_count) if tt != t))
                if current_max < best_max or (current_max == best_max and (t < best_route or (t == best_route and pos < best_pos))):
                    best_max = current_max
                    best_route = t
                    best_pos = pos
        # insert at best position
        route = routes[best_route]
        routes[best_route] = route[:best_pos] + [cust] + route[best_pos:]
    
    best_routes = [list(r) for r in routes]
    best_max_dist = compute_max_distance()
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass
    
    # --- Improvement ---
    max_iter = n * truck_count * 2  # bounded
    for iteration in range(max_iter):
        improved = False
        # 2-opt within each route
        for t in range(truck_count):
            route = routes[t]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                    new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                    delta = new_edges - old_edges
                    if delta < -1e-9:
                        new_dist = route_distance(route) + delta
                        # compute new max for whole solution
                        new_max = max(new_dist, max(route_distance(routes[tt]) for tt in range(truck_count) if tt != t))
                        if new_max < best_max_dist - 1e-9:
                            # apply reversal
                            routes[t] = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            best_max_dist = new_max
                            best_routes = [list(r) for r in routes]
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 2-opt* exchange tails between two routes
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for t2 in range(t1+1, truck_count):
                route2 = routes[t2]
                if len(route2) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        # tail1: route1[i+1:len(route1)-1] (excluding last depot)
                        # tail2: route2[j+1:len(route2)-1]
                        # new edges: (route1[i], route2[j+1]) and (route2[j], route1[i+1])
                        old1 = distance_matrix[route1[i]][route1[i+1]]
                        old2 = distance_matrix[route2[j]][route2[j+1]]
                        new1 = distance_matrix[route1[i]][route2[j+1]]
                        new2 = distance_matrix[route2[j]][route1[i+1]]
                        delta1 = new1 - old1
                        delta2 = new2 - old2
                        # new distances for the two routes (others unchanged)
                        new_dist1 = route_distance(route1) + delta1
                        new_dist2 = route_distance(route2) + delta2
                        other_max = max(route_distance(routes[tt]) for tt in range(truck_count) if tt != t1 and tt != t2)
                        new_max = max(new_dist1, new_dist2, other_max)
                        if new_max < best_max_dist - 1e-9:
                            # apply exchange: swap tails
                            tail1 = route1[i+1:-1]  # without last depot
                            tail2 = route2[j+1:-1]
                            new_route1 = route1[:i+1] + tail2 + [depot]
                            new_route2 = route2[:j+1] + tail1 + [depot]
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            best_max_dist = new_max
                            best_routes = [list(r) for r in routes]
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best_routes