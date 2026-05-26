import numpy as np

INF = 1e12

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # trivial case: each customer on its own route
    if truck_count >= len(customers):
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # Initialize each customer as a separate route
    routes = [[0, c, 0] for c in customers]
    
    # Merge using Clarke-Wright savings until truck_count routes remain
    while len(routes) > truck_count:
        best_saving = -INF
        best_pair = None
        best_order = 0
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                ri = routes[i]
                rj = routes[j]
                if len(ri) <= 2 or len(rj) <= 2:
                    continue
                last_i = ri[-2]
                first_i = ri[1]
                last_j = rj[-2]
                first_j = rj[1]
                s1 = distance_matrix[0][last_i] + distance_matrix[0][first_j] - distance_matrix[last_i][first_j]
                s2 = distance_matrix[0][last_j] + distance_matrix[0][first_i] - distance_matrix[last_j][first_i]
                if s1 > best_saving:
                    best_saving = s1
                    best_pair = (i, j)
                    best_order = 0
                if s2 > best_saving:
                    best_saving = s2
                    best_pair = (i, j)
                    best_order = 1
        if best_pair is None:
            break
        i, j = best_pair
        if best_order == 0:
            new_route = routes[i][:-1] + routes[j][1:]
        else:
            new_route = routes[j][:-1] + routes[i][1:]
        if i < j:
            del routes[j]
            del routes[i]
        else:
            del routes[i]
            del routes[j]
        routes.append(new_route)
    
    # Compute initial distances and report
    dists = [route_distance(r, distance_matrix) for r in routes]
    best_routes = [list(r) for r in routes]
    best_max = max(dists)
    report_best_vrp(best_routes)
    
    max_restarts = 5
    for restart in range(max_restarts):
        max_iter = n * truck_count
        improved_overall = False
        for _ in range(max_iter):
            dists = [route_distance(r, distance_matrix) for r in routes]
            max_dist = max(dists)
            if max_dist < best_max - 1e-12:
                best_max = max_dist
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            max_idx = dists.index(max_dist)
            improved = False
            # Relocate moves from longest route
            if len(routes[max_idx]) > 2:
                for pos in range(1, len(routes[max_idx])-1):
                    cust = routes[max_idx][pos]
                    new_max_route = routes[max_idx][:pos] + routes[max_idx][pos+1:]
                    new_max_dist = route_distance(new_max_route, distance_matrix)
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = routes[other_idx]
                        for insert_pos in range(1, len(other_route)):
                            new_other_route = other_route[:insert_pos] + [cust] + other_route[insert_pos:]
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            if new_max < max_dist - 1e-12:
                                routes[max_idx] = new_max_route
                                routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            # If no relocate improvement, try swap
            if not improved and len(routes[max_idx]) > 2:
                for other_idx in range(truck_count):
                    if other_idx == max_idx or len(routes[other_idx]) <= 2:
                        continue
                    for pos_max in range(1, len(routes[max_idx])-1):
                        cust_a = routes[max_idx][pos_max]
                        for pos_other in range(1, len(routes[other_idx])-1):
                            cust_b = routes[other_idx][pos_other]
                            new_max_route = routes[max_idx].copy()
                            new_max_route[pos_max] = cust_b
                            new_max_dist = route_distance(new_max_route, distance_matrix)
                            new_other_route = routes[other_idx].copy()
                            new_other_route[pos_other] = cust_a
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            if new_max < max_dist - 1e-12:
                                routes[max_idx] = new_max_route
                                routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            # If still not improved, try intra-route 2-opt on each route
            if not improved:
                for r_idx in range(truck_count):
                    if len(routes[r_idx]) <= 3:
                        continue
                    route = routes[r_idx]
                    best_route = route[:]
                    best_len = route_distance(route, distance_matrix)
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_len = route_distance(new_route, distance_matrix)
                            if new_len < best_len - 1e-12:
                                best_len = new_len
                                best_route = new_route[:]
                    if best_route != route:
                        routes[r_idx] = best_route
                        dists[r_idx] = best_len
                        improved = True
                        break
            if improved:
                improved_overall = True
            else:
                break
        if not improved_overall:
            # Perturbation: relocate one customer from longest route to another tour
            if restart < max_restarts - 1:  # not last restart
                dists = [route_distance(r, distance_matrix) for r in routes]
                max_idx = dists.index(max(dists))
                # Find a customer to move (first interior customer)
                if len(routes[max_idx]) > 2:
                    cust_pos = 1  # first interior
                    cust = routes[max_idx][cust_pos]
                    new_max_route = routes[max_idx][:cust_pos] + routes[max_idx][cust_pos+1:]
                    # Insert into another route at best improving position (but we just perturb, so insert at position 1 of the shortest route?)
                    other_idx = min((i for i in range(truck_count) if i != max_idx), key=lambda i: dists[i])
                    other_route = routes[other_idx]
                    # insert at the position that gives smallest increase in other route distance
                    best_other = None
                    best_inc = INF
                    for ins_pos in range(1, len(other_route)):
                        new_other = other_route[:ins_pos] + [cust] + other_route[ins_pos:]
                        new_other_dist = route_distance(new_other, distance_matrix)
                        if new_other_dist - dists[other_idx] < best_inc - 1e-12:
                            best_inc = new_other_dist - dists[other_idx]
                            best_other = new_other
                    if best_other is not None:
                        routes[max_idx] = new_max_route
                        routes[other_idx] = best_other
        else:
            break
    # Final report and return
    report_best_vrp(best_routes)
    return best_routes