import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    unassigned = set(range(1, n))
    routes = [[depot, depot] for _ in range(truck_count)]
    
    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def best_insertion(cust, route):
        best_cost = float('inf')
        best_pos = -1
        for pos in range(1, len(route)):
            i = route[pos-1]
            j = route[pos]
            cost = distance_matrix[i, cust] + distance_matrix[cust, j] - distance_matrix[i, j]
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        return best_cost, best_pos
    
    # Cheapest insertion construction
    while unassigned:
        best_cost = float('inf')
        best_cust = -1
        best_route_idx = -1
        best_pos = -1
        for cust in unassigned:
            for r_idx, route in enumerate(routes):
                cost, pos = best_insertion(cust, route)
                if cost < best_cost or (cost == best_cost and cust < best_cust):
                    best_cost = cost
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
        route = routes[best_route_idx]
        route.insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)
    
    # Local search improvement
    n_customers = n - 1
    max_iters = 2 * n_customers
    for _ in range(max_iters):
        improved = False
        # Inter-route relocate
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            customers_in_route = route[1:-1]
            for cust in customers_in_route:
                new_route = [x for x in route if x != cust]
                for other_idx, other_route in enumerate(routes):
                    if other_idx == r_idx:
                        continue
                    cost, pos = best_insertion(cust, other_route)
                    candidate_routes = [list(r) for r in routes]
                    candidate_routes[r_idx] = new_route
                    other_new = list(other_route)
                    other_new.insert(pos, cust)
                    candidate_routes[other_idx] = other_new
                    dists = [route_dist(r) for r in candidate_routes]
                    new_max = max(dists)
                    if new_max < best_max:
                        best_max = new_max
                        routes = candidate_routes
                        report_best_vrp(routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Intra-route 2-opt
            for r_idx, route in enumerate(routes):
                if len(route) <= 4:
                    continue
                n_nodes = len(route)
                best_imp = False
                for i in range(1, n_nodes-2):
                    for j in range(i+1, n_nodes-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_dist(new_route) < route_dist(route):
                            routes[r_idx] = new_route
                            best_imp = True
                            break
                    if best_imp:
                        break
                if best_imp:
                    dists = [route_dist(r) for r in routes]
                    new_max = max(dists)
                    if new_max < best_max:
                        best_max = new_max
                        report_best_vrp(routes)
                    improved = True
                    break
        if not improved:
            break
    
    result = []
    for r in routes:
        if len(r) <= 2:
            result.append([0, 0])
        else:
            if r[0] != 0:
                r.insert(0, 0)
            if r[-1] != 0:
                r.append(0)
            result.append(r)
    while len(result) < truck_count:
        result.append([0, 0])
    return result