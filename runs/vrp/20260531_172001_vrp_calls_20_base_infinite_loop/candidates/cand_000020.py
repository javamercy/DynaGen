import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    def route_length(route):
        if len(route) <= 2:
            return 0.0
        total = 0.0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    # Initialize routes
    routes = [[0, 0] for _ in range(truck_count)]
    
    # Regret-insertion construction
    unassigned = set(customers)
    while unassigned:
        best_cust = None
        best_regret = -1e9
        best_cost = None
        best_route_idx = None
        best_pos = None
        for cust in unassigned:
            insert_costs = []
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = (distance_matrix[prev, cust] + 
                            distance_matrix[cust, nxt] - 
                            distance_matrix[prev, nxt])
                    insert_costs.append((cost, r_idx, pos))
            insert_costs.sort(key=lambda x: x[0])
            best = insert_costs[0][0]
            second = insert_costs[1][0] if len(insert_costs) > 1 else best + 1e9
            regret = second - best
            if (regret > best_regret or 
                (abs(regret - best_regret) < 1e-12 and best_cost is not None and best > best_cost) or
                (abs(regret - best_regret) < 1e-12 and best_cost is not None and abs(best - best_cost) < 1e-12 and cust < best_cust)):
                best_regret = regret
                best_cost = best
                best_cust = cust
                best_route_idx = insert_costs[0][1]
                best_pos = insert_costs[0][2]
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)
    
    # Improvement phase
    current_max = max(route_length(r) for r in routes)
    report_best_vrp(routes)
    max_iter = n * truck_count
    iteration = 0
    improved = True
    
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        
        # 1) Inter-route relocate: move a customer from longest route to another to reduce max
        lengths = [route_length(r) for r in routes]
        max_idx = max(range(truck_count), key=lambda i: lengths[i])
        max_route = routes[max_idx]
        if len(max_route) > 3:
            for cust in max_route[1:-1]:
                new_max_route = [0] + [c for c in max_route[1:-1] if c != cust] + [0]
                new_max_len = route_length(new_max_route)
                for r_idx in range(truck_count):
                    if r_idx == max_idx:
                        continue
                    other_route = routes[r_idx]
                    for pos in range(1, len(other_route)):
                        new_other = other_route[:pos] + [cust] + other_route[pos:]
                        new_other_len = route_length(new_other)
                        new_max = max(new_max_len, new_other_len,
                                      *[route_length(routes[i]) for i in range(truck_count) if i not in (max_idx, r_idx)])
                        if new_max < current_max - 1e-12:
                            routes[max_idx] = new_max_route
                            routes[r_idx] = new_other
                            current_max = new_max
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if improved:
            continue
        
        # 2) Intra-route 2-opt on each route
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for k in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    new_len = route_length(new_route)
                    if new_len < route_length(route) - 1e-12:
                        old_len = route_length(route)
                        routes[r_idx] = new_route
                        new_max = max(route_length(r) for r in routes)
                        if new_max < current_max - 1e-12:
                            current_max = new_max
                            report_best_vrp(routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return routes