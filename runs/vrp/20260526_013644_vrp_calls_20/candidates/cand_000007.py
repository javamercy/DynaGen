import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    unassigned = set(customers)
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
    
    # Regret insertion construction
    while unassigned:
        best_regret = -1
        best_cust = -1
        best_route_idx = -1
        best_pos = -1
        best_cost_for_cust = float('inf')
        for cust in unassigned:
            costs = []
            for r_idx, route in enumerate(routes):
                cost, pos = best_insertion(cust, route)
                costs.append((cost, r_idx, pos))
            costs.sort(key=lambda x: x[0])
            if len(costs) == 1:
                regret = costs[0][0] * 2
            else:
                regret = costs[1][0] - costs[0][0]
            if regret > best_regret or (regret == best_regret and costs[0][0] > best_cost_for_cust):
                best_regret = regret
                best_cust = cust
                best_cost_for_cust = costs[0][0]
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]
            elif regret == best_regret and costs[0][0] == best_cost_for_cust:
                if cust < best_cust:
                    best_cust = cust
                    best_route_idx = costs[0][1]
                    best_pos = costs[0][2]
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)
    
    n_cust = n - 1
    max_iters = 5 * n_cust  # bounded
    for _ in range(max_iters):
        improved = False
        # Compute current route distances
        dists = [route_dist(r) for r in routes]
        max_idx = max(range(truck_count), key=lambda i: (dists[i], i))  # smallest index tie
        max_route = routes[max_idx]
        # Inter-route relocate from max route to any other route (best move)
        best_move = None
        best_new_max = best_max
        for cust_idx_in_route in range(1, len(max_route)-1):  # skip depots
            cust = max_route[cust_idx_in_route]
            new_max_route = max_route[:cust_idx_in_route] + max_route[cust_idx_in_route+1:]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                cost, pos = best_insertion(cust, other_route)
                new_other = list(other_route)
                new_other.insert(pos, cust)
                # Compute distances for affected routes
                new_max_route_dist = route_dist(new_max_route)
                new_other_dist = route_dist(new_other)
                # Other distances unchanged
                other_dists = [dists[i] for i in range(truck_count) if i != max_idx and i != other_idx]
                cand_max = max([new_max_route_dist, new_other_dist] + other_dists)
                if cand_max < best_new_max:
                    best_new_max = cand_max
                    best_move = (max_idx, other_idx, cust_idx_in_route, pos, cust, new_max_route, new_other)
        if best_move is not None:
            max_idx, other_idx, cust_idx_in_route, pos, cust, new_max_route, new_other = best_move
            routes[max_idx] = new_max_route
            routes[other_idx] = new_other
            best_max = best_new_max
            improved = True
            report_best_vrp(routes)
        else:
            # Intra-route 2-opt on max route
            if len(max_route) > 3:
                best_2opt = None
                best_2opt_dist = route_dist(max_route)
                for i in range(1, len(max_route)-2):
                    for j in range(i+1, len(max_route)-1):
                        new_route = max_route[:i] + max_route[i:j+1][::-1] + max_route[j+1:]
                        new_dist = route_dist(new_route)
                        # Only consider if it reduces max route distance
                        if new_dist < best_2opt_dist:
                            best_2opt_dist = new_dist
                            best_2opt = (i, j, new_route)
                if best_2opt is not None:
                    i, j, new_route = best_2opt
                    routes[max_idx] = new_route
                    # Update best_max
                    dists = [route_dist(r) for r in routes]
                    new_max = max(dists)
                    if new_max < best_max:
                        best_max = new_max
                        improved = True
                        report_best_vrp(routes)
        if not improved:
            break
    # Ensure exactly truck_count routes, each [0,0] if empty
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