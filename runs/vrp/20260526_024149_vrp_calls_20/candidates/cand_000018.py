import numpy as np

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    num_customers = len(customers)
    
    if truck_count >= num_customers:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # Initialization: each customer as a separate route
    routes = [[0, c, 0] for c in customers]
    
    # Clarke-Wright merging
    while len(routes) > truck_count:
        best_saving = -1e9
        best_pair = None
        best_order = 0
        for i in range(len(routes)):
            r_i = routes[i]
            for j in range(i+1, len(routes)):
                r_j = routes[j]
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
    
    report_best_vrp(routes)
    
    dists = [route_distance(r, distance_matrix) for r in routes]
    best_max = max(dists)
    best_routes = [list(r) for r in routes]
    
    max_iter = 2 * n
    for _ in range(max_iter):
        current_max = max(dists)
        longest_idx = max(range(len(dists)), key=lambda i: dists[i])
        
        # Intra-route 2-opt on longest route
        longest_route = routes[longest_idx]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(longest_route)-2):
                for k in range(i+1, len(longest_route)-1):
                    # swap edges (i-1,i) and (k,k+1) with (i-1,k) and (i,k+1)
                    if k - i == 1:
                        continue
                    old_cost = distance_matrix[longest_route[i-1], longest_route[i]] + distance_matrix[longest_route[k], longest_route[k+1]]
                    new_cost = distance_matrix[longest_route[i-1], longest_route[k]] + distance_matrix[longest_route[i], longest_route[k+1]]
                    if new_cost < old_cost - 1e-9:
                        longest_route = longest_route[:i] + longest_route[i:k+1][::-1] + longest_route[k+1:]
                        improved = True
            if improved:
                routes[longest_idx] = longest_route
                dists[longest_idx] = route_distance(longest_route, distance_matrix)
                if dists[longest_idx] < best_max - 1e-9:
                    best_max = dists[longest_idx]
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
        
        # Relocation moves from longest route
        best_improvement = 0
        best_move = None
        for pos_in_longest, cust in enumerate(routes[longest_idx]):
            if cust == 0:
                continue
            prev_node = routes[longest_idx][pos_in_longest-1]
            next_node = routes[longest_idx][pos_in_longest+1]
            removal_saving = distance_matrix[prev_node, cust] + distance_matrix[cust, next_node] - distance_matrix[prev_node, next_node]
            new_long_len = dists[longest_idx] - removal_saving
            for r_idx, route in enumerate(routes):
                if r_idx == longest_idx:
                    continue
                for pos in range(1, len(route)):
                    if route[pos] == cust or route[pos-1] == cust:
                        continue
                    insertion_cost = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                    new_target_len = dists[r_idx] + insertion_cost
                    new_max = max(new_long_len, new_target_len)
                    if new_max < current_max - 1e-9:
                        improvement = current_max - new_max
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (longest_idx, pos_in_longest, r_idx, pos, cust)
        
        if best_move is not None:
            longest_idx, pos_in_longest, r_idx, pos, cust = best_move
            routes[longest_idx].pop(pos_in_longest)
            routes[r_idx].insert(pos, cust)
            dists[longest_idx] = route_distance(routes[longest_idx], distance_matrix)
            dists[r_idx] = route_distance(routes[r_idx], distance_matrix)
            new_max = max(dists)
            if new_max < best_max - 1e-9:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
        else:
            break
    
    return best_routes