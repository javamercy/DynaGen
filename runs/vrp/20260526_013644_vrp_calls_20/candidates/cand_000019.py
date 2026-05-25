import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    unassigned = set(customers)
    routes = [[depot, depot] for _ in range(truck_count)]
    
    def route_dist(route):
        d = 0.0
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
    
    # Regret insertion construction
    while unassigned:
        best_regret = -1.0
        best_cust = -1
        best_route_idx = -1
        best_pos = -1
        best_cost_for_cust = float('inf')
        for cust in list(unassigned):
            costs = []
            for r_idx, route in enumerate(routes):
                cost, pos = best_insertion(cust, route)
                costs.append((cost, r_idx, pos))
            costs.sort(key=lambda x: x[0])
            if len(costs) == 1:
                regret = 1e9
            else:
                regret = costs[1][0] - costs[0][0]
            if (regret > best_regret or
                (regret == best_regret and costs[0][0] > best_cost_for_cust) or
                (regret == best_regret and costs[0][0] == best_cost_for_cust and cust < best_cust)):
                best_regret = regret
                best_cust = cust
                best_cost_for_cust = costs[0][0]
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]
        route = routes[best_route_idx]
        route.insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    
    # Initial best
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)
    
    n_customers = n - 1
    max_iters = 2 * n_customers
    for iteration in range(max_iters):
        improved = False
        # Inter-route relocate (best improvement on max)
        best_move = None
        best_new_max = best_max
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for cust in route[1:-1]:
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
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = (r_idx, cust, other_idx, pos, candidate_routes)
        if best_move is not None:
            routes = best_move[4]
            best_max = best_new_max
            improved = True
            report_best_vrp(routes)
        else:
            # Intra-route 2-opt (best improvement on max)
            for r_idx, route in enumerate(routes):
                if len(route) <= 4:
                    continue
                best_imp = None
                best_dist = route_dist(route)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_dist(new_route)
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_imp = (i, j, new_route)
                if best_imp is not None:
                    routes[r_idx] = best_imp[2]
                    improved = True
                    dists = [route_dist(r) for r in routes]
                    new_max = max(dists)
                    if new_max < best_max:
                        best_max = new_max
                        report_best_vrp(routes)
            if not improved:
                # Cross-route 2-opt* (best improvement on max)
                best_move = None
                best_new_max = best_max
                for r1_idx in range(truck_count):
                    for r2_idx in range(r1_idx+1, truck_count):
                        r1 = routes[r1_idx]
                        r2 = routes[r2_idx]
                        if len(r1) <= 3 or len(r2) <= 3:
                            continue
                        # Try all possible cut points (positions after customer, indices 1..len-1)
                        for i in range(1, len(r1)-1):
                            for j in range(1, len(r2)-1):
                                new_r1 = r1[:i] + r2[j:]
                                new_r2 = r2[:j] + r1[i:]
                                candidate_routes = [list(r) for r in routes]
                                candidate_routes[r1_idx] = new_r1
                                candidate_routes[r2_idx] = new_r2
                                dists = [route_dist(r) for r in candidate_routes]
                                new_max = max(dists)
                                if new_max < best_new_max:
                                    best_new_max = new_max
                                    best_move = (r1_idx, r2_idx, i, j, candidate_routes)
                if best_move is not None:
                    routes = best_move[4]
                    best_max = best_new_max
                    improved = True
                    report_best_vrp(routes)
                else:
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