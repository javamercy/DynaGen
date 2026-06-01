import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # Helper functions
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes) if routes else 0.0
    
    def copy_routes(routes):
        return [list(r) for r in routes]
    
    # Step 1: Savings-based construction
    # Start with each customer as a separate route
    routes = [[0, c, 0] for c in customers]
    best_routes = copy_routes(routes)
    best_max = max_route_distance(routes)
    report_best_vrp(best_routes)
    
    # Merge until we have exactly truck_count routes
    iteration = 0
    max_iter = n * n  # Safety bound
    while len(routes) > truck_count and iteration < max_iter:
        iteration += 1
        # Collect all possible merges (combine routes by connecting end to start)
        # For each pair of routes, possible endpoints: last customer of first route (depot removed) and first customer of second route (depot removed)
        best_merge = None
        best_new_max = float('inf')
        best_pair = None
        
        # For efficiency, precompute route distances and customer lists
        route_dists = [route_distance(r) for r in routes]
        route_ends = [(r[-2], r[1]) for r in routes]  # last customer before depot, first customer after depot
        
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue
                # Check if merge is possible: routes i and j
                # Extract customers (excluding depots)
                cust_i = routes[i][1:-1]
                cust_j = routes[j][1:-1]
                if len(cust_i) == 0 or len(cust_j) == 0:
                    continue
                # end of i is cust_i[-1], start of j is cust_j[0]
                # New route: depot + cust_i + cust_j + depot
                new_route_cust = cust_i + cust_j
                new_route = [0] + new_route_cust + [0]
                new_dist = route_distance(new_route)
                # Compute new max route distance if this merge is performed
                other_dists = [route_dists[k] for k in range(len(routes)) if k != i and k != j]
                new_max = max(other_dists + [new_dist]) if other_dists else new_dist
                
                # Choose merge with smallest new_max; tie-break by sum of route indices, then customer indices
                if new_max < best_new_max or (new_max == best_new_max and (i+j < best_pair[0]+best_pair[1] if best_pair else True)):
                    best_new_max = new_max
                    best_pair = (i, j)
                    best_merge = new_route
        
        if best_merge is None:
            break
        # Perform merge: replace routes i and j with best_merge (but we need to remove them and add new)
        i, j = best_pair
        # Ensure i < j to avoid index issues after removal
        if i > j:
            i, j = j, i
        # Remove j first (higher index), then i
        routes.pop(j)
        routes.pop(i)
        routes.append(best_merge)
        # Update best solution if needed
        current_max = max_route_distance(routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = copy_routes(routes)
            report_best_vrp(best_routes)
    
    # Now we have exactly truck_count routes (may be less if not enough merges, but initial has n-1 >= truck_count typically)
    # If still more, we might have to merge with tying, but unlikely with n=100, truck_count=6
    # Step 2: Tabu search improvement
    max_iter2 = n * n
    tabu_tenure = int(np.sqrt(n)) + 1
    tabu_dict = {}  # key: (type, details), value: remaining tenure
    
    current_routes = copy_routes(routes)
    current_max = max_route_distance(current_routes)
    if current_max < best_max:
        best_max = current_max
        best_routes = copy_routes(current_routes)
        report_best_vrp(best_routes)
    
    for iteration2 in range(max_iter2):
        # Generate all moves (relocate and swap) that are not tabu or that improve best
        # We'll use first-improving: find first move that improves current_max
        improved = False
        # List of moves: store (type, params, new_routes, new_max)
        # We'll sort moves by new_max, then tie-break deterministic
        candidate_moves = []
        
        # Relocate moves: move one customer from its route to another route
        for i, route in enumerate(current_routes):
            if len(route) <= 3:
                continue
            for pos in range(1, len(route)-1):
                cust = route[pos]
                for j, other_route in enumerate(current_routes):
                    if i == j:
                        continue
                    for other_pos in range(1, len(other_route)):
                        # Build new routes
                        new_self = route[:pos] + route[pos+1:]
                        new_other = other_route[:other_pos] + [cust] + other_route[other_pos:]
                        new_routes = [list(r) for r in current_routes]
                        new_routes[i] = new_self
                        new_routes[j] = new_other
                        new_max = max_route_distance(new_routes)
                        # Check tabu: (cust, from_route_i, to_route_j) for relocate
                        tabu_key = ('relocate', cust, i, j)
                        if tabu_key in tabu_dict and tabu_dict[tabu_key] > 0:
                            # Tabu, but allow if it improves best?
                            # Typically tabu restriction overridden by aspiration: if new_max < best_max, accept
                            if new_max >= best_max:
                                continue
                        candidate_moves.append(('relocate', i, pos, j, other_pos, new_routes, new_max, tabu_key))
        
        # Swap moves: swap two customers from different routes
        for i, route_i in enumerate(current_routes):
            if len(route_i) <= 3:
                continue
            for pos_i in range(1, len(route_i)-1):
                cust_i = route_i[pos_i]
                for j, route_j in enumerate(current_routes):
                    if i >= j:
                        continue
                    if len(route_j) <= 3:
                        continue
                    for pos_j in range(1, len(route_j)-1):
                        cust_j = route_j[pos_j]
                        # Build new routes
                        new_route_i = route_i[:pos_i] + [cust_j] + route_i[pos_i+1:]
                        new_route_j = route_j[:pos_j] + [cust_i] + route_j[pos_j+1:]
                        new_routes = [list(r) for r in current_routes]
                        new_routes[i] = new_route_i
                        new_routes[j] = new_route_j
                        new_max = max_route_distance(new_routes)
                        tabu_key = ('swap', cust_i, i, cust_j, j)
                        if tabu_key in tabu_dict and tabu_dict[tabu_key] > 0:
                            if new_max >= best_max:
                                continue
                        candidate_moves.append(('swap', i, pos_i, j, pos_j, new_routes, new_max, tabu_key))
        
        if not candidate_moves:
            break
        # Sort moves by new_max, then by some deterministic tie (e.g., cust IDs, route indices)
        # For simplicity, we sort by new_max, then by the tuple of parameters
        def move_key(m):
            # m: (type, ..., new_routes, new_max, tabu_key)
            # We use new_max, then type, then route indices, then positions, then customer IDs
            if m[0] == 'relocate':
                return (m[6], m[0], m[1], m[3], m[2], m[4])  # new_max, type, from_route, to_route, pos_from, pos_to
            else:  # swap
                return (m[6], m[0], m[1], m[3], m[2], m[4])
        candidate_moves.sort(key=move_key)
        
        # Select the best move
        best_move = candidate_moves[0]
        typ, i, pos_i, j, pos_j, new_routes, new_max, tabu_key = best_move
        
        # Execute the move
        current_routes = new_routes
        current_max = new_max
        if current_max < best_max:
            best_max = current_max
            best_routes = copy_routes(current_routes)
            report_best_vrp(best_routes)
        
        # Update tabu list: add the move's tabu key with tenure, decrement all
        tabu_dict[tabu_key] = tabu_tenure
        # Decrement all tabu tenures
        keys_to_remove = []
        for k in tabu_dict:
            tabu_dict[k] -= 1
            if tabu_dict[k] <= 0:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del tabu_dict[k]
    
    return best_routes