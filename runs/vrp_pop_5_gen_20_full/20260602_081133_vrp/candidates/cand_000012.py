import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # Regret-2 construction
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(customers)
    # Insert each customer using regret-2
    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    def insertion_cost(route, pos, cust):
        # cost increase if insert cust at pos in route (pos between 0 and len(route)-1)
        return distance_matrix[route[pos]][cust] + distance_matrix[cust][route[pos+1]] - distance_matrix[route[pos]][route[pos+1]]
    
    def best_insertion(cust, route):
        best_cost = float('inf')
        best_pos = -1
        for pos in range(1, len(route)):
            cost = insertion_cost(route, pos, cust)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        return best_pos, best_cost
    
    while unassigned:
        # For each unassigned customer, compute best cost per route and regret
        regrets = []
        for cust in unassigned:
            costs = []
            for r in range(truck_count):
                _, cost = best_insertion(cust, routes[r])
                costs.append(cost)
            costs.sort()
            regret = costs[1] - costs[0] if len(costs) >= 2 else costs[0]
            regrets.append((regret, cust, costs[0]))
        # Sort by regret descending, then by cost ascending, then by customer index
        regrets.sort(key=lambda x: (-x[0], x[1], x[2]))
        _, cust, _ = regrets[0]
        # insert into route with minimal cost
        best_route_idx = -1
        best_cost = float('inf')
        best_pos = -1
        for r in range(truck_count):
            pos, cost = best_insertion(cust, routes[r])
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
                best_route_idx = r
        routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
        unassigned.remove(cust)
    
    # Compute initial max distance
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)
    
    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)
    try:
        report_best_vrp(best_routes)
    except:
        pass
    
    # Multi-start local search
    n_customers = n - 1
    max_restarts = n_customers  # bounded
    for restart in range(max_restarts):
        # Perturb: randomly relocate a few customers (up to 2% of n, min 1)
        num_perturb = max(1, int(0.02 * n_customers))
        for _ in range(num_perturb):
            # Pick a random customer from a random route
            r_idx = np.random.randint(0, truck_count)
            route = routes[r_idx]
            if len(route) <= 2:
                continue
            cust_idx = np.random.randint(1, len(route)-1)
            cust = route[cust_idx]
            # Remove
            route.pop(cust_idx)
            # Insert into random route at best position
            target_r = np.random.randint(0, truck_count)
            pos, _ = best_insertion(cust, routes[target_r])
            routes[target_r] = routes[target_r][:pos] + [cust] + routes[target_r][pos:]
        
        # Local search: improve max distance
        improved = True
        max_iter = n_customers * truck_count
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            current_max = max_route_distance(routes)
            # Identify the longest route
            longest_idx = max(range(truck_count), key=lambda i: route_distance(routes[i]))
            longest_route = routes[longest_idx]
            
            # Inter-route relocate: move a customer from longest to any other
            for cust in longest_route[1:-1]:
                for r_idx in range(truck_count):
                    if r_idx == longest_idx:
                        continue
                    # Remove cust from longest
                    new_longest = [c for c in longest_route if c != cust or c == 0]
                    if len(new_longest) == 1:
                        new_longest = [0, 0]
                    # Insert into best position of target
                    pos, _ = best_insertion(cust, routes[r_idx])
                    new_target = routes[r_idx][:pos] + [cust] + routes[r_idx][pos:]
                    new_routes = routes.copy()
                    new_routes[longest_idx] = new_longest
                    new_routes[r_idx] = new_target
                    new_max = max_route_distance(new_routes)
                    if new_max < best_max:
                        best_routes = new_routes
                        best_max = new_max
                        improved = True
                        try:
                            report_best_vrp(best_routes)
                        except:
                            pass
                        break
                if improved:
                    break
            if improved:
                routes = [list(r) for r in best_routes]
                continue
            
            # Inter-route Or-opt: move a segment (1-3 customers) from longest to another
            for seg_len in [1, 2, 3]:
                if improved:
                    break
                if len(longest_route) - 2 < seg_len:
                    continue
                for start in range(1, len(longest_route) - seg_len):
                    if improved:
                        break
                    seg = longest_route[start:start+seg_len]
                    new_longest = longest_route[:start] + longest_route[start+seg_len:]
                    for r_idx in range(truck_count):
                        if r_idx == longest_idx:
                            continue
                        # Try inserting seg as a block in every position of target route
                        target_route = routes[r_idx]
                        for pos in range(1, len(target_route)):
                            new_target = target_route[:pos] + seg + target_route[pos:]
                            new_routes = routes.copy()
                            new_routes[longest_idx] = new_longest
                            new_routes[r_idx] = new_target
                            new_max = max_route_distance(new_routes)
                            if new_max < best_max:
                                best_routes = new_routes
                                best_max = new_max
                                improved = True
                                try:
                                    report_best_vrp(best_routes)
                                except:
                                    pass
                                break
                        if improved:
                            break
                
            if improved:
                routes = [list(r) for r in best_routes]
                continue
            
            # Intra-route 2-opt on the longest route
            if not improved:
                route = longest_route
                for i in range(1, len(route)-2):
                    if improved:
                        break
                    for j in range(i+1, len(route)-1):
                        if j-i == 1:
                            continue
                        old_dist = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new_dist = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        if new_dist < old_dist:
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_routes = routes.copy()
                            new_routes[longest_idx] = new_route
                            new_max = max_route_distance(new_routes)
                            if new_max < best_max:
                                best_routes = new_routes
                                best_max = new_max
                                improved = True
                                try:
                                    report_best_vrp(best_routes)
                                except:
                                    pass
                                break
                    
            if improved:
                routes = [list(r) for r in best_routes]
                continue
            
            # Cross-route swap: swap two customers between longest and another
            for cust_long in longest_route[1:-1]:
                if improved:
                    break
                for r_idx in range(truck_count):
                    if r_idx == longest_idx:
                        continue
                    other_route = routes[r_idx]
                    for cust_other in other_route[1:-1]:
                        new_longest = [c if c != cust_long else cust_other for c in longest_route]
                        # Ensure depots remain
                        new_longest[0] = 0
                        new_longest[-1] = 0
                        new_other = [c if c != cust_other else cust_long for c in other_route]
                        new_other[0] = 0
                        new_other[-1] = 0
                        new_routes = routes.copy()
                        new_routes[longest_idx] = new_longest
                        new_routes[r_idx] = new_other
                        new_max = max_route_distance(new_routes)
                        if new_max < best_max:
                            best_routes = new_routes
                            best_max = new_max
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except:
                                pass
                            break
                    if improved:
                        break
            if improved:
                routes = [list(r) for r in best_routes]
        
        # After local search, update routes for next restart if better
        max_cur = max_route_distance(routes)
        if max_cur < best_max:
            best_routes = [list(r) for r in routes]
            best_max = max_cur
            try:
                report_best_vrp(best_routes)
            except:
                pass
    
    return best_routes