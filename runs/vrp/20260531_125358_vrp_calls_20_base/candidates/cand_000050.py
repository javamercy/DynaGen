def solve_vrp(distance_matrix, truck_count):
    import random
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = []
        for i in range(truck_count):
            if i < len(customers):
                routes.append([0, customers[i], 0])
            else:
                routes.append([0,0])
        return routes
    def route_length(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    def two_opt(route, max_iter=10):
        route = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_length(new_route) < route_length(route):
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
        return route
    def balance_routes(routes, lengths):
        improved = True
        while improved:
            improved = False
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_overall_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_length(new_max_route)
                min_route = routes[min_idx]
                best_insertion_len = float('inf')
                best_pos = -1
                for p in range(1, len(min_route)):
                    new_min_route = min_route[:p] + [cust] + min_route[p:]
                    l = route_length(new_min_route)
                    if l < best_insertion_len:
                        best_insertion_len = l
                        best_pos = p
                new_min_route = min_route[:best_pos] + [cust] + min_route[best_pos:]
                new_min_len = route_length(new_min_route)
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_overall_reduction:
                    best_overall_reduction = reduction
                    best_cust = (cust, best_pos)
            if best_cust is not None:
                cust, best_insert_pos = best_cust
                new_max = [node for node in max_route if node != cust]
                min_route = routes[min_idx]
                new_min = min_route[:best_insert_pos] + [cust] + min_route[best_insert_pos:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_length(new_max)
                lengths[min_idx] = route_length(new_min)
                improved = True
                report_best_vrp(routes)
            else:
                break
        return routes, lengths
    best_routes = None
    best_max = float('inf')
    num_restarts = min(10, max(1, n//10))
    for _ in range(num_restarts):
        # Random assignment: for each customer, assign to a random truck
        assignment = [random.randrange(truck_count) for _ in customers]
        routes = [[] for _ in range(truck_count)]
        for idx, cust in enumerate(customers):
            routes[assignment[idx]].append(cust)
        # Build routes starting and ending at 0
        routes_list = []
        for i in range(truck_count):
            if not routes[i]:
                routes_list.append([0,0])
            else:
                # randomize order? no, then apply nearest neighbor? For diversity, we keep random order then improve
                random.shuffle(routes[i])
                route = [0] + routes[i] + [0]
                routes_list.append(route)
        # 2-opt on each route
        for i in range(truck_count):
            if len(routes_list[i]) > 2:
                routes_list[i] = two_opt(routes_list[i], max_iter=len(routes_list[i])-2)
        lengths = [route_length(r) for r in routes_list]
        # Balancing
        routes_list, lengths = balance_routes(routes_list, lengths)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes_list]
            report_best_vrp(best_routes)
        # Perturbation loop: stronger perturbation
        max_perturb = min(100, n * truck_count)
        current_routes = [r[:] for r in routes_list]
        current_lengths = lengths[:]
        for _ in range(max_perturb):
            # Select a random route with at least 2 customers
            usable = [i for i, r in enumerate(current_routes) if len(r) > 3]  # at least one internal customer
            if not usable or len(usable) < 1:
                break
            src = random.choice(usable)
            # Remove a random segment of length 1..max(1, len(route)//2)
            route = current_routes[src]
            max_len = max(1, (len(route)-2)//2)
            seg_len = random.randint(1, min(max_len, len(route)-2))
            start = random.randint(1, len(route)-1-seg_len)
            segment = route[start:start+seg_len]
            remaining = route[:start] + route[start+seg_len:]
            # Reinsert segment customers into other routes at random positions
            new_routes = current_routes[:]
            new_routes[src] = remaining
            for cust in segment:
                dst = random.randrange(truck_count)
                # Choose a random insertion position in destination route
                dst_route = new_routes[dst]
                if len(dst_route) == 2:
                    ins_pos = 1
                else:
                    ins_pos = random.randint(1, len(dst_route)-1)
                new_route = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                new_routes[dst] = new_route
            # Apply 2-opt to affected routes (src and all dst)
            affected = set([src] + [random.randrange(truck_count) for _ in segment])
            for i in affected:
                if i < truck_count and len(new_routes[i]) > 2:
                    new_routes[i] = two_opt(new_routes[i], max_iter=5)
            new_lengths = [route_length(r) for r in new_routes]
            # Balance
            new_routes, new_lengths = balance_routes(new_routes, new_lengths)
            new_max = max(new_lengths)
            # Accept if better or with small probability (simulated annealing? but we want exploration, accept always?)
            if new_max < best_max:
                best_max = new_max
                best_routes = [r[:] for r in new_routes]
                report_best_vrp(best_routes)
            # Always move to new state
            current_routes = new_routes
            current_lengths = new_lengths
    return best_routes