import numpy as np

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def two_opt(route, dm):
    """Improve a single route using 2-opt, returns improved route."""
    improved = True
    best_route = route[:]
    best_dist = route_distance(best_route, dm)
    while improved:
        improved = False
        for i in range(1, len(best_route)-2):
            for j in range(i+1, len(best_route)-1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_dist = route_distance(new_route, dm)
                if new_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    return best_route

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
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
        best_saving = -1e9
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
    
    # Improvement phase with restarts
    max_restarts = 3
    max_iter = n * truck_count
    for restart in range(max_restarts):
        # Apply intra-route 2-opt on all routes
        for idx in range(truck_count):
            if len(routes[idx]) > 2:
                routes[idx] = two_opt(routes[idx], distance_matrix)
        dists = [route_distance(r, distance_matrix) for r in routes]
        curr_max = max(dists)
        if curr_max < best_max - 1e-9:
            best_max = curr_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        
        # Local search: relocate and swap moves focusing on max route
        for iteration in range(max_iter):
            dists = [route_distance(r, distance_matrix) for r in routes]
            max_dist = max(dists)
            if max_dist < best_max - 1e-9:
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
                            if new_max < max_dist - 1e-9:
                                routes[max_idx] = new_max_route
                                routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            # If no relocate improvement, try swap moves
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
                            if new_max < max_dist - 1e-9:
                                routes[max_idx] = new_max_route
                                routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        
        # After local search, if this is the last restart, skip perturbation
        if restart == max_restarts - 1:
            break
        # Perturbation: move few customers from longest to shortest route
        dists = [route_distance(r, distance_matrix) for r in routes]
        max_dist = max(dists)
        min_dist = min(dists)
        # find indices of longest and shortest routes (deterministic: first by order)
        max_idx = dists.index(max_dist)
        # find shortest route (by distance, tie by smallest index)
        min_idx = None
        for idx in range(truck_count):
            if idx != max_idx:
                if min_idx is None or dists[idx] < dists[min_idx]:
                    min_idx = idx
        if max_idx is None or min_idx is None:
            break
        # Remove up to 2 customers from longest route (if feasible)
        num_to_move = min(2, len(routes[max_idx])-2)
        if num_to_move <= 0:
            continue
        # Remove the first `num_to_move` customers (positions 1..num_to_move)
        removed_customers = []
        for _ in range(num_to_move):
            # Always remove the customer at position 1 (after depot) to keep indices stable
            if len(routes[max_idx]) <= 2:
                break
            removed_customers.append(routes[max_idx].pop(1))
        # Insert them into shortest route at best positions (evaluate all orders and positions? simple: insert each at best spot greedily)
        for cust in reversed(removed_customers):  # maintain order of removal might not matter
            best_insert_pos = 1
            best_insert_dist = float('inf')
            for pos in range(1, len(routes[min_idx])):
                new_route = routes[min_idx][:pos] + [cust] + routes[min_idx][pos:]
                new_dist = route_distance(new_route, distance_matrix)
                if new_dist < best_insert_dist:
                    best_insert_dist = new_dist
                    best_insert_pos = pos
            routes[min_idx].insert(best_insert_pos, cust)
        # After perturbation, update distances and best if improved
        dists = [route_distance(r, distance_matrix) for r in routes]
        curr_max = max(dists)
        if curr_max < best_max - 1e-9:
            best_max = curr_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    
    report_best_vrp(best_routes)
    return best_routes