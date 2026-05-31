import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    
    def route_length(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    # Min-max greedy insertion
    unassigned = set(customers)
    while unassigned:
        best_customer = None
        best_route_idx = None
        best_pos = None
        best_max_after = float('inf')
        best_cost = None
        for cust in unassigned:
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    next_node = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, next_node] - distance_matrix[prev, next_node]
                    new_len = route_length(route) + cost
                    other_lens = [route_length(r) for i, r in enumerate(routes) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    if new_max < best_max_after or (new_max == best_max_after and (best_cost is None or cost > best_cost)):
                        best_max_after = new_max
                        best_customer = cust
                        best_route_idx = r_idx
                        best_pos = pos
                        best_cost = cost
                    elif new_max == best_max_after and cost == best_cost and cust < best_customer:
                        best_customer = cust
                        best_route_idx = r_idx
                        best_pos = pos
                        best_cost = cost
        route = routes[best_route_idx]
        route.insert(best_pos, best_customer)
        unassigned.remove(best_customer)
    
    report_best_vrp(routes)  # report initial solution
    
    current_max = max(route_length(r) for r in routes)
    max_iter = n * truck_count
    for _ in range(max_iter):
        improved = False
        best_delta = 0
        best_move = None
        
        # Inter-route relocate: consider moving any customer to any other route at any position
        for from_idx in range(truck_count):
            from_route = routes[from_idx]
            if len(from_route) <= 2:
                continue
            for cust in from_route[1:-1]:
                new_from = [x for x in from_route if x != cust]
                new_from_len = route_length(new_from)
                for to_idx in range(truck_count):
                    if to_idx == from_idx:
                        continue
                    to_route = routes[to_idx]
                    for pos in range(1, len(to_route)):
                        new_to = to_route[:pos] + [cust] + to_route[pos:]
                        new_to_len = route_length(new_to)
                        other_lens = [route_length(r) for i, r in enumerate(routes) if i not in (from_idx, to_idx)]
                        new_max = max(new_from_len, new_to_len, *other_lens)
                        delta = current_max - new_max
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (cust, from_idx, to_idx, pos)
        
        # Intra-route 2-opt: consider reversing subsequences in each route
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for k in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    new_len = route_length(new_route)
                    old_len = route_length(route)
                    if new_len < old_len:
                        other_lens = [route_length(r) for j, r in enumerate(routes) if j != r_idx]
                        new_max = max(new_len, *other_lens)
                        delta = current_max - new_max
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (r_idx, i, k, new_route)  # store new route for 2-opt
        
        if best_delta > 0 and best_move is not None:
            if len(best_move) == 4 and isinstance(best_move[0], int) and best_move[3] is list:
                # 2-opt move
                r_idx, i, k, new_route = best_move
                routes[r_idx] = new_route
            else:
                # relocate move
                cust, from_idx, to_idx, pos = best_move
                routes[from_idx] = [x for x in routes[from_idx] if x != cust]
                routes[to_idx].insert(pos, cust)
            current_max = current_max - best_delta
            improved = True
            report_best_vrp(routes)
        
        if not improved:
            break
    
    return routes