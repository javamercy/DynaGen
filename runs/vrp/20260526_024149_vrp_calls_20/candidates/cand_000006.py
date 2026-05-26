import numpy as np

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    num_customers = len(customers)
    
    # Edge case: more trucks than customers
    if truck_count >= num_customers:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # Initialization: each customer as a separate route
    routes = [[0, c, 0] for c in customers]
    
    # Merge until we have exactly truck_count routes
    while len(routes) > truck_count:
        best_saving = -1e9
        best_pair = None
        best_order = 0
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                r_i = routes[i]
                r_j = routes[j]
                # Routes have at least depot nodes; length >= 2
                last_i = r_i[-2]
                first_i = r_i[1]
                last_j = r_j[-2]
                first_j = r_j[1]
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
    
    # Report initial construction
    report_best_vrp(routes)
    
    # Compute initial distances
    dists = [route_distance(r, distance_matrix) for r in routes]
    best_max = max(dists)
    best_routes = [list(r) for r in routes]
    
    # Local search: best-improvement relocate
    max_iter = 2 * n
    for _ in range(max_iter):
        current_max = max(dists)
        best_improvement = 0
        best_move = None
        
        # For each customer
        for cust in range(1, n):
            # Find current route and position
            curr_route_idx = -1
            curr_pos = -1
            for r_idx, route in enumerate(routes):
                if cust in route:
                    curr_route_idx = r_idx
                    curr_pos = route.index(cust)
                    break
            if curr_route_idx == -1:
                continue
            curr_route = routes[curr_route_idx]
            # Removal cost
            prev_node = curr_route[curr_pos-1]
            next_node = curr_route[curr_pos+1]
            removal_saving = distance_matrix[prev_node, cust] + distance_matrix[cust, next_node] - distance_matrix[prev_node, next_node]
            new_curr_len = dists[curr_route_idx] - removal_saving
            
            # Try inserting into other routes (including same route? skip to avoid cycles)
            for r_idx, route in enumerate(routes):
                if r_idx == curr_route_idx:
                    continue
                for pos in range(1, len(route)):  # positions 1..len(route)-1, insert between route[pos-1] and route[pos]
                    if route[pos] == cust or route[pos-1] == cust:
                        continue
                    insertion_cost = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                    new_len = dists[r_idx] + insertion_cost
                    # Compute new max
                    new_dists = dists.copy()
                    new_dists[curr_route_idx] = new_curr_len
                    new_dists[r_idx] = new_len
                    new_max = max(new_dists)
                    if new_max < current_max - 1e-9:
                        improvement = current_max - new_max
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (curr_route_idx, curr_pos, r_idx, pos, cust)
        
        if best_move is not None:
            curr_route_idx, curr_pos, r_idx, pos, cust = best_move
            # Remove from current route
            routes[curr_route_idx].pop(curr_pos)
            # Insert into target route
            routes[r_idx].insert(pos, cust)
            # Update distances
            dists = [route_distance(r, distance_matrix) for r in routes]
            new_max = max(dists)
            if new_max < best_max - 1e-9:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
        else:
            break
    
    return best_routes