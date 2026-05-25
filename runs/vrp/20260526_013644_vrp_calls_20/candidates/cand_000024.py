import numpy as np


def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    # ---- Regret construction ----
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
        route = routes[best_route_idx]
        route.insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)
    
    # ---- Simulated Annealing Local Search ----
    current_routes = [list(r) for r in best_routes]
    current_max = best_max
    n_customers = n - 1
    max_iters = 4 * n_customers  # finite bound
    T_start = 0.3 * best_max  # initial temperature
    T_end = 0.01
    T = T_start
    
    for it in range(max_iters):
        if T < T_end:
            break
        # Generate neighbor: inter-route relocate or 2-opt or split-merge
        # We'll try each neighborhood in order, accept if improves or via SA
        improved = False
        # ---- Inter-route relocate ----
        for r_idx, route in enumerate(current_routes):
            if len(route) <= 3:
                continue
            cust_list = route[1:-1]
            for cust in cust_list:
                new_route_src = [x for x in route if x != cust]
                for other_idx, other_route in enumerate(current_routes):
                    if other_idx == r_idx:
                        continue
                    cost, pos = best_insertion(cust, other_route)
                    candidate_routes = [list(r) for r in current_routes]
                    candidate_routes[r_idx] = new_route_src
                    other_new = list(other_route)
                    other_new.insert(pos, cust)
                    candidate_routes[other_idx] = other_new
                    dists = [route_dist(r) for r in candidate_routes]
                    new_max = max(dists)
                    delta = new_max - current_max
                    if delta < 0 or np.random.random() < np.exp(-delta / T):
                        current_routes = candidate_routes
                        current_max = new_max
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [list(r) for r in current_routes]
                            report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            T *= 0.99
            continue
        # ---- Intra-route 2-opt ----
        for r_idx, route in enumerate(current_routes):
            if len(route) <= 4:
                continue
            n_nodes = len(route)
            for i in range(1, n_nodes-2):
                for j in range(i+1, n_nodes-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    candidate_routes = [list(r) for r in current_routes]
                    candidate_routes[r_idx] = new_route
                    dists = [route_dist(r) for r in candidate_routes]
                    new_max = max(dists)
                    delta = new_max - current_max
                    if delta < 0 or np.random.random() < np.exp(-delta / T):
                        current_routes = candidate_routes
                        current_max = new_max
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [list(r) for r in current_routes]
                            report_best_vrp(best_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            T *= 0.99
            continue
        # ---- Split/Merge ----
        # Pick two distinct routes with at least 2 customers each (after depots)
        for r1_idx in range(truck_count):
            r1 = current_routes[r1_idx]
            if len(r1) <= 3:  # depot + one customer
                continue
            for r2_idx in range(truck_count):
                if r2_idx <= r1_idx:
                    continue
                r2 = current_routes[r2_idx]
                if len(r2) <= 3:
                    continue
                # For each possible split point in r1 (excluding first and last depot)
                for split in range(1, len(r1)-1):
                    # segment from r1[split] to r1[-2] (inclusive) to be moved
                    if split >= len(r1)-1:
                        continue
                    segment = r1[split:-1]  # list of customers
                    # Remove segment from r1
                    new_r1 = r1[:split] + [r1[-1]]
                    # Insert segment into r2 at each possible position (excluding last depot)
                    for pos in range(1, len(r2)):
                        new_r2 = r2[:pos] + segment + r2[pos:]
                        candidate_routes = [list(r) for r in current_routes]
                        candidate_routes[r1_idx] = new_r1
                        candidate_routes[r2_idx] = new_r2
                        dists = [route_dist(r) for r in candidate_routes]
                        new_max = max(dists)
                        delta = new_max - current_max
                        if delta < 0 or np.random.random() < np.exp(-delta / T):
                            current_routes = candidate_routes
                            current_max = new_max
                            if new_max < best_max:
                                best_max = new_max
                                best_routes = [list(r) for r in current_routes]
                                report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            T *= 0.99
        else:
            T *= 0.98  # cool down even if no improvement
    
    # Ensure exactly truck_count routes, each starting/ending with 0
    result = []
    for r in best_routes:
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