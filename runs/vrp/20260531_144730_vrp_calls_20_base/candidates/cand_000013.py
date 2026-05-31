import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    
    # Helper to compute route distance
    def route_distance(route):
        if len(route) < 2:
            return 0.0
        return sum(distance_matrix[u][v] for u, v in zip(route[:-1], route[1:]))
    
    # 1. Initial construction: each customer as a separate route
    routes = {}
    for i in range(1, n):
        routes[i-1] = [0, i, 0]
    next_route_id = n - 1
    # Add empty routes if needed
    for _ in range(max(0, truck_count - (n-1))):
        routes[next_route_id] = [0, 0]
        next_route_id += 1
    
    # Map customer to route id
    customer_to_route = {i: i-1 for i in range(1, n)}
    
    # 2. Compute savings
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))
    
    # 3. Merge routes using savings until truck_count routes remain
    merge_attempts = 0
    max_merge_iter = n * n  # safety bound
    while len(routes) > truck_count and merge_attempts < max_merge_iter:
        merge_attempts += 1
        merged = False
        for s, i, j in savings:
            if len(routes) <= truck_count:
                break
            ri = customer_to_route.get(i)
            rj = customer_to_route.get(j)
            if ri is None or rj is None or ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            # skip if either route is empty (only depot)
            if len(route_i) <= 2 or len(route_j) <= 2:
                continue
            # Determine endpoints
            start_i = route_i[1]
            end_i = route_i[-2]
            start_j = route_j[1]
            end_j = route_j[-2]
            # Possible orientations (i_pos, j_pos) where pos is 's' or 'e'
            orientations = []
            if i == start_i and j == start_j:
                orientations.append(('s', 's'))
            if i == start_i and j == end_j:
                orientations.append(('s', 'e'))
            if i == end_i and j == start_j:
                orientations.append(('e', 's'))
            if i == end_i and j == end_j:
                orientations.append(('e', 'e'))
            if not orientations:
                continue
            # Evaluate each orientation and pick best (minimum total distance increase)
            best_new_route = None
            best_increase = float('inf')
            for i_pos, j_pos in orientations:
                # Build merged route
                if i_pos == 's' and j_pos == 's':
                    # Reverse route_i so that i becomes end
                    rev_i = [0] + route_i[-2:0:-1] + [0]
                    new_route = rev_i[:-1] + route_j[1:]
                elif i_pos == 's' and j_pos == 'e':
                    new_route = route_i[:-1] + route_j[1:]
                elif i_pos == 'e' and j_pos == 's':
                    rev_j = [0] + route_j[-2:0:-1] + [0]
                    new_route = route_i[:-1] + rev_j[1:]
                else:  # i_pos == 'e' and j_pos == 'e'
                    rev_i = [0] + route_i[-2:0:-1] + [0]
                    rev_j = [0] + route_j[-2:0:-1] + [0]
                    new_route = rev_i[:-1] + rev_j[1:]
                # Compute cost increase
                old_cost = route_distance(route_i) + route_distance(route_j)
                new_cost = route_distance(new_route)
                increase = new_cost - old_cost
                if increase < best_increase:
                    best_increase = increase
                    best_new_route = new_route
            # Perform merge with best orientation
            if best_new_route is not None:
                # Remove old routes and add new
                del routes[ri]
                del routes[rj]
                routes[next_route_id] = best_new_route
                # Update customer to route mapping
                for cust in best_new_route[1:-1]:
                    customer_to_route[cust] = next_route_id
                next_route_id += 1
                merged = True
                break  # restart savings scan after each merge
        if not merged:
            # If no valid merge from savings, force merge two longest routes
            # (but this is rare, so simple fallback)
            # Pick routes with smallest number of customers (excluding empty) and merge
            non_empty = [rid for rid in routes if len(routes[rid]) > 2]
            if len(non_empty) >= 2:
                # Choose the two with smallest route ids for determinism
                non_empty.sort()
                ri, rj = non_empty[0], non_empty[1]
                route_i = routes[ri]
                route_j = routes[rj]
                # Simple end-to-start merge
                new_route = route_i[:-1] + route_j[1:]
                del routes[ri]
                del routes[rj]
                routes[next_route_id] = new_route
                for cust in new_route[1:-1]:
                    customer_to_route[cust] = next_route_id
                next_route_id += 1
            else:
                break
    
    # Ensure exactly truck_count routes: if too few, add empty; if too many, merge further
    route_list = list(routes.values())
    while len(route_list) < truck_count:
        route_list.append([0, 0])
    # If still more than truck_count, merge arbitrarily (should not happen)
    while len(route_list) > truck_count:
        # Merge last two non-empty routes (if any empty, just remove)
        non_empty_indices = [idx for idx, r in enumerate(route_list) if len(r) > 2]
        if len(non_empty_indices) >= 2:
            # Merge first two (by index)
            i, j = non_empty_indices[0], non_empty_indices[1]
            route_list[i] = route_list[i][:-1] + route_list[j][1:]
            del route_list[j]
        else:
            # Remove an empty route if exists
            empty_idx = next((idx for idx, r in enumerate(route_list) if len(r)==2), None)
            if empty_idx is not None:
                del route_list[empty_idx]
            else:
                break
    
    # Report initial solution
    report_best_vrp(route_list)
    
    # 4. Improvement phase: intensive local search focusing on max distance
    # Initial max distance
    current_max = max(route_distance(r) for r in route_list)
    best_routes = [list(r) for r in route_list]
    best_max = current_max
    
    # Bounds for loops
    max_iter = 2 * n * n  # overall iterations
    iter_count = 0
    improved = True
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        # First, intra-route 2-opt on each route
        for idx in range(len(route_list)):
            route = route_list[idx]
            if len(route) <= 3:
                continue
            best_dist = route_distance(route)
            best_route = route[:]
            for start in range(1, len(route)-2):
                for end in range(start+1, len(route)-1):
                    new_route = route[:start] + route[start:end+1][::-1] + route[end+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_route = new_route
                        improved = True
            route_list[idx] = best_route
        
        # Update max after 2-opt
        new_max = max(route_distance(r) for r in route_list)
        if new_max < best_max:
            best_max = new_max
            best_routes = [list(r) for r in route_list]
            report_best_vrp(best_routes)
        
        # Inter-route relocate and swap with best-improvement strategy
        # First, collect all possible moves and compute effect on max distance
        # We'll do a full scan of all relocate and swap moves across pairs
        # To bound complexity, we limit by n^3? Actually O(K^2 * n^2) where K=truck_count <= n
        n_routes = len(route_list)
        best_move = None  # tuple (type, params, new_routes)
        best_new_max = float('inf')
        
        # For each pair of routes
        for i in range(n_routes):
            for j in range(i+1, n_routes):
                route_i = route_list[i]
                route_j = route_list[j]
                # Skip if both empty or too short?
                if len(route_i) <= 2 and len(route_j) <= 2:
                    continue
                # Relocate from i to j
                for ci_idx in range(1, len(route_i)-1):
                    cust = route_i[ci_idx]
                    new_route_i = route_i[:ci_idx] + route_i[ci_idx+1:]
                    # Find best insertion position in route_j
                    best_j_dist = float('inf')
                    best_j_route = None
                    for ins in range(1, len(route_j)):
                        candidate_j = route_j[:ins] + [cust] + route_j[ins:]
                        dist_j = route_distance(candidate_j)
                        if dist_j < best_j_dist:
                            best_j_dist = dist_j
                            best_j_route = candidate_j
                    # Compute new max
                    dist_i = route_distance(new_route_i)
                    # Max of all routes: we need to compute only changed ones, but others unchanged
                    new_max_candidate = max(dist_i, best_j_dist)
                    # Actually need to consider all routes, but others are same as current max? 
                    # We'll compute properly:
                    candidate_routes = list(route_list)
                    candidate_routes[i] = new_route_i
                    candidate_routes[j] = best_j_route
                    candidate_max = max(route_distance(r) for r in candidate_routes)
                    if candidate_max < best_new_max:
                        best_new_max = candidate_max
                        best_move = ('relocate', i, j, ci_idx, None, candidate_routes)
                
                # Swap between i and j
                for ci_idx in range(1, len(route_i)-1):
                    for cj_idx in range(1, len(route_j)-1):
                        cust_i = route_i[ci_idx]
                        cust_j = route_j[cj_idx]
                        new_route_i = route_i[:ci_idx] + [cust_j] + route_i[ci_idx+1:]
                        new_route_j = route_j[:cj_idx] + [cust_i] + route_j[cj_idx+1:]
                        candidate_routes = list(route_list)
                        candidate_routes[i] = new_route_i
                        candidate_routes[j] = new_route_j
                        candidate_max = max(route_distance(r) for r in candidate_routes)
                        if candidate_max < best_new_max:
                            best_new_max = candidate_max
                            best_move = ('swap', i, j, ci_idx, cj_idx, candidate_routes)
        
        # Apply best move if it improves
        if best_move is not None and best_new_max < current_max:
            route_list = best_move[5]
            current_max = best_new_max
            if best_new_max < best_max:
                best_max = best_new_max
                best_routes = [list(r) for r in route_list]
                report_best_vrp(best_routes)
            improved = True
        
        # If no improvement, try perturbation: relocate random customer? But we want deterministic. Instead, do a simple shuffle: perform a random relocate move (but deterministic by taking first valid). Since we want deterministic, we can take first move that does not worsen max (or even slightly worsen). But for exploitation focus, we avoid perturbation, so we just break.
    
    # Return best found routes
    return best_routes