import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    # Construction: min-max insertion
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = set(customers)
    
    while unassigned:
        best_cust = None
        best_max = float('inf')
        best_move = None
        for cust in list(unassigned):
            best_new_max = float('inf')
            best_t = -1
            best_pos = -1
            for t in range(truck_count):
                route = routes[t]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_distance(new_route)
                    new_max = new_dist
                    for k in range(truck_count):
                        if k != t:
                            new_max = max(new_max, route_dists[k])
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_t = t
                        best_pos = pos
            if best_new_max < best_max or (best_new_max == best_max and cust < best_cust):
                best_max = best_new_max
                best_cust = cust
                best_move = (best_t, best_pos)
        t, pos = best_move
        routes[t] = routes[t][:pos] + [best_cust] + routes[t][pos:]
        route_dists[t] = route_distance(routes[t])
        unassigned.remove(best_cust)
    
    best_max = max(route_dists)
    best_routes = [list(r) for r in routes]
    
    def update_best(routes):
        nonlocal best_max, best_routes
        m = max(route_distance(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    
    update_best(routes)
    
    # Hill climbing: relocate and swap
    max_iter = (n - 1) * truck_count * 2
    for _ in range(max_iter):
        best_move = None
        best_new_max = float('inf')
        best_tie = None
        
        # Relocate
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for i in range(1, len(route1) - 1):
                cust = route1[i]
                new_route1 = route1[:i] + route1[i+1:]
                dist1_new = route_distance(new_route1)
                for t2 in range(truck_count):
                    if t1 == t2:
                        continue
                    route2 = routes[t2]
                    for j in range(1, len(route2)):
                        new_route2 = route2[:j] + [cust] + route2[j:]
                        dist2_new = route_distance(new_route2)
                        new_max = max(dist1_new, dist2_new)
                        for k in range(truck_count):
                            if k != t1 and k != t2:
                                new_max = max(new_max, route_dists[k])
                        if new_max < best_new_max or (new_max == best_new_max and best_tie is None):
                            tie = (new_max, 0, t1, i, t2, j)
                            if best_tie is None or tie < best_tie:
                                best_new_max = new_max
                                best_move = ('relocate', t1, i, t2, j, cust)
                                best_tie = tie
        # Swap
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for i in range(1, len(route1) - 1):
                cust1 = route1[i]
                for t2 in range(t1 + 1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    for j in range(1, len(route2) - 1):
                        cust2 = route2[j]
                        new_route1 = route1[:i] + [cust2] + route1[i+1:]
                        new_route2 = route2[:j] + [cust1] + route2[j+1:]
                        dist1_new = route_distance(new_route1)
                        dist2_new = route_distance(new_route2)
                        new_max = max(dist1_new, dist2_new)
                        for k in range(truck_count):
                            if k != t1 and k != t2:
                                new_max = max(new_max, route_dists[k])
                        if new_max < best_new_max or (new_max == best_new_max and best_tie is None):
                            tie = (new_max, 1, t1, i, t2, j)
                            if best_tie is None or tie < best_tie:
                                best_new_max = new_max
                                best_move = ('swap', t1, i, t2, j, cust1, cust2)
                                best_tie = tie
        if best_move is None or best_new_max >= max(route_dists):
            break
        # Apply move
        if best_move[0] == 'relocate':
            _, t1, i, t2, j, cust = best_move
            routes[t1] = routes[t1][:i] + routes[t1][i+1:]
            routes[t2] = routes[t2][:j] + [cust] + routes[t2][j:]
            route_dists[t1] = route_distance(routes[t1])
            route_dists[t2] = route_distance(routes[t2])
        else:  # swap
            _, t1, i, t2, j, cust1, cust2 = best_move
            routes[t1][i] = cust2
            routes[t2][j] = cust1
            route_dists[t1] = route_distance(routes[t1])
            route_dists[t2] = route_distance(routes[t2])
        update_best(routes)
    
    return best_routes