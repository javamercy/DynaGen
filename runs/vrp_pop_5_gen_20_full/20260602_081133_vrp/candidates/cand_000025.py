import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    # farthest-first insertion construction
    depot_distances = distance_matrix[0, 1:]
    order = sorted(range(1, n), key=lambda x: -depot_distances[x-1])
    routes = [[0, 0] for _ in range(truck_count)]
    for cust in order:
        best_route = -1
        best_pos = -1
        best_inc = float('inf')
        for r in range(truck_count):
            route = routes[r]
            # if route is empty (only depot), length 2
            for pos in range(1, len(route)):
                inc = (distance_matrix[route[pos-1]][cust] +
                       distance_matrix[cust][route[pos]] -
                       distance_matrix[route[pos-1]][route[pos]])
                if inc < best_inc:
                    best_inc = inc
                    best_route = r
                    best_pos = pos
                elif inc == best_inc and r < best_route:
                    best_route = r
                    best_pos = pos
        # insert
        route = routes[best_route]
        routes[best_route] = route[:best_pos] + [cust] + route[best_pos:]
    
    def route_dist(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    
    def total_dist_of_routes(rts):
        return [route_dist(r) for r in rts]
    
    def compute_max(rts):
        return max(route_dist(r) for r in rts)
    
    def copy_routes(rts):
        return [list(r) for r in rts]
    
    best_routes = copy_routes(routes)
    best_max = compute_max(routes)
    try:
        report_best_vrp(best_routes)
    except:
        pass
    
    # local search
    improved = True
    max_iter = n * truck_count
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        # intra-route 2-opt
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    old_dist = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                    new_dist = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                    if new_dist < old_dist - 1e-12:
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        routes[r] = new_route
                        improved = True
                        new_max = compute_max(routes)
                        if new_max < best_max - 1e-12:
                            best_routes = copy_routes(routes)
                            best_max = new_max
                            try:
                                report_best_vrp(best_routes)
                            except:
                                pass
                        # restart because route changed
                        break
                if improved:
                    break
        if improved:
            continue
        # inter-route relocate
        for r_from in range(truck_count):
            for cust in routes[r_from][1:-1]:
                for r_to in range(truck_count):
                    if r_to == r_from:
                        continue
                    route_from = routes[r_from]
                    route_to = routes[r_to]
                    for pos in range(1, len(route_to)):
                        new_from = route_from[1:-1]
                        new_from.remove(cust)
                        new_from = [0] + new_from + [0]
                        if len(new_from) == 1:
                            new_from = [0, 0]
                        new_to = route_to[:pos] + [cust] + route_to[pos:]
                        new_routes = copy_routes(routes)
                        new_routes[r_from] = new_from
                        new_routes[r_to] = new_to
                        new_max = compute_max(new_routes)
                        if new_max < best_max - 1e-12:
                            routes = new_routes
                            best_routes = copy_routes(routes)
                            best_max = new_max
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # inter-route swap
        for r1 in range(truck_count):
            for c1 in routes[r1][1:-1]:
                for r2 in range(r1+1, truck_count):
                    for c2 in routes[r2][1:-1]:
                        # swap c1 and c2
                        new_r1 = [0] + [c2 if x==c1 else x for x in routes[r1][1:-1]] + [0]
                        new_r2 = [0] + [c1 if x==c2 else x for x in routes[r2][1:-1]] + [0]
                        new_routes = copy_routes(routes)
                        new_routes[r1] = new_r1
                        new_routes[r2] = new_r2
                        new_max = compute_max(new_routes)
                        if new_max < best_max - 1e-12:
                            routes = new_routes
                            best_routes = copy_routes(routes)
                            best_max = new_max
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # inter-route 2-opt* (cross exchange)
        for r1 in range(truck_count):
            for r2 in range(r1+1, truck_count):
                route1 = routes[r1]
                route2 = routes[r2]
                if len(route1) <= 2 or len(route2) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        # new route1: route1[:i] + route2[j:]
                        # new route2: route2[:j] + route1[i:]
                        new1 = route1[:i] + route2[j:]
                        new2 = route2[:j] + route1[i:]
                        # check feasibility (depots at ends)
                        if new1[0] != 0 or new1[-1] != 0 or new2[0] != 0 or new2[-1] != 0:
                            continue
                        # check all customers present exactly once
                        set1 = set(new1[1:-1])
                        set2 = set(new2[1:-1])
                        if len(set1) + len(set2) != (len(route1)-2) + (len(route2)-2):
                            continue
                        if len(set1 & set2) > 0:
                            continue
                        new_routes = copy_routes(routes)
                        new_routes[r1] = new1
                        new_routes[r2] = new2
                        new_max = compute_max(new_routes)
                        if new_max < best_max - 1e-12:
                            routes = new_routes
                            best_routes = copy_routes(routes)
                            best_max = new_max
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best_routes