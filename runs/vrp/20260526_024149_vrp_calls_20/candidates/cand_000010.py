import numpy as np

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    num_customers = len(customers)
    
    # If more trucks than customers, assign each customer to a truck and fill rest with empty routes
    if truck_count >= num_customers:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # Initialization: each customer as a separate route
    routes = [[0, c, 0] for c in customers]
    
    # Merge until exactly truck_count routes using Clarke-Wright savings
    while len(routes) > truck_count:
        best_saving = -np.inf
        best_pair = None
        best_order = 0
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                r_i = routes[i]
                r_j = routes[j]
                # Skip routes that are just depot (empty) - shouldn't happen here
                if len(r_i) <= 2 or len(r_j) <= 2:
                    continue
                last_i = r_i[-2]
                first_i = r_i[1]
                last_j = r_j[-2]
                first_j = r_j[1]
                s1 = distance_matrix[0, last_i] + distance_matrix[0, first_j] - distance_matrix[last_i, first_j]
                s2 = distance_matrix[0, last_j] + distance_matrix[0, first_i] - distance_matrix[last_j, first_i]
                if s1 > best_saving:
                    best_saving = s1
                    best_pair = (i, j)
                    best_order = 0
                if s2 > best_saving:
                    best_saving = s2
                    best_pair = (i, j)
                    best_order = 1
        if best_pair is None:
            # Fallback: if no merge possible, break (should not happen)
            break
        i, j = best_pair
        if best_order == 0:
            new_route = routes[i][:-1] + routes[j][1:]
        else:
            new_route = routes[j][:-1] + routes[i][1:]
        # Remove routes in reverse order to preserve indices
        if i < j:
            del routes[j]
            del routes[i]
        else:
            del routes[i]
            del routes[j]
        routes.append(new_route)
    
    # If after merging we have fewer routes than truck_count, add empty routes (should not happen)
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    report_best_vrp(routes)
    
    # Improvement: steepest descent relocation of single customers to minimize max distance
    max_iter = num_customers * truck_count  # finite bound
    for _ in range(max_iter):
        dists = [route_distance(r, distance_matrix) for r in routes]
        max_dist = max(dists)
        min_dist = min(dists)
        if max_dist == min_dist:
            break
        max_idx = dists.index(max_dist)
        best_improvement = 0.0
        best_move = None
        route_max = routes[max_idx]
        if len(route_max) <= 2:
            break  # no customer to move
        for pos_cust in range(1, len(route_max) - 1):
            customer = route_max[pos_cust]
            for other_idx, other_route in enumerate(routes):
                if other_idx == max_idx:
                    continue
                for insert_pos in range(1, len(other_route)):
                    new_route_max = route_max[:pos_cust] + route_max[pos_cust + 1:]
                    new_other = other_route[:insert_pos] + [customer] + other_route[insert_pos:]
                    new_dists = dists.copy()
                    new_dists[max_idx] = route_distance(new_route_max, distance_matrix)
                    new_dists[other_idx] = route_distance(new_other, distance_matrix)
                    new_max = max(new_dists)
                    if new_max < max_dist - 1e-9:
                        improvement = max_dist - new_max
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (max_idx, pos_cust, other_idx, insert_pos)
        if best_move is not None:
            max_idx, pos_cust, other_idx, insert_pos = best_move
            customer = routes[max_idx][pos_cust]
            routes[max_idx] = routes[max_idx][:pos_cust] + routes[max_idx][pos_cust + 1:]
            routes[other_idx] = routes[other_idx][:insert_pos] + [customer] + routes[other_idx][insert_pos:]
            report_best_vrp(routes)
        else:
            break
    
    # Ensure exactly truck_count routes (should already be)
    routes = routes[:truck_count]  # safety truncate
    report_best_vrp(routes)
    return routes