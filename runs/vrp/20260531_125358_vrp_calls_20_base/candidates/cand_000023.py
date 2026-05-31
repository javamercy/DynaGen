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
            best_improvement = 0
            best_i, best_j = -1, -1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    old_len = route_length(route)
                    new_len = route_length(new_route)
                    improvement = old_len - new_len
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_i, best_j = i, j
            if best_improvement > 0:
                route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                improved = True
        return route
    
    def balance_routes(routes, lengths):
        improved = True
        while improved:
            improved = False
            max_idx = max(range(truck_count), key=lambda i: (lengths[i], -i))
            min_idx = min(range(truck_count), key=lambda i: (lengths[i], i))
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
    
    def shake(routes, lengths, alpha=0.05):
        n_cust = sum(len(r)-2 for r in routes if len(r)>2)
        num_remove = max(1, int(n_cust * alpha))
        all_customers = []
        for i, r in enumerate(routes):
            if len(r) > 2:
                for c in r[1:-1]:
                    all_customers.append((i, c))
        if len(all_customers) < num_remove:
            return routes, lengths
        random.shuffle(all_customers)
        removed = []
        for idx in range(num_remove):
            route_idx, cust = all_customers[idx]
            routes[route_idx] = [node for node in routes[route_idx] if node != cust]
            removed.append(cust)
        # reinsert removed customers greedily with random tie-breaking
        for cust in removed:
            best_route = 0
            best_increase = float('inf')
            best_pos = 1
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for p in range(1, len(route)):
                    new_len = route_length(route[:p] + [cust] + route[p:])
                    old_len = route_length(route)
                    increase = new_len - old_len
                    if increase < best_increase or (increase == best_increase and random.random() < 0.5):
                        best_increase = increase
                        best_route = r_idx
                        best_pos = p
            routes[best_route] = routes[best_route][:best_pos] + [cust] + routes[best_route][best_pos:]
        lengths = [route_length(r) for r in routes]
        return routes, lengths
    
    best_routes = None
    best_max = float('inf')
    num_restarts = min(15, max(1, n//5))
    for _ in range(num_restarts):
        seeds = random.sample(customers, min(truck_count, len(customers)))
        clusters = [[] for _ in range(truck_count)]
        for i, s in enumerate(seeds):
            clusters[i].append(s)
        remaining = [c for c in customers if c not in seeds]
        for cust in remaining:
            best_dist = distance_matrix[cust][seeds[0]]
            best_clusters = [0]
            for i in range(1, truck_count):
                d = distance_matrix[cust][seeds[i]]
                if d < best_dist:
                    best_dist = d
                    best_clusters = [i]
                elif d == best_dist:
                    best_clusters.append(i)
            chosen = random.choice(best_clusters) if len(best_clusters)>1 else best_clusters[0]
            clusters[chosen].append(cust)
        routes = []
        for i in range(truck_count):
            if not clusters[i]:
                routes.append([0,0])
            else:
                unvisited = set(clusters[i])
                route = [0]
                current = 0
                while unvisited:
                    if random.random() < 0.2:
                        next_node = random.choice(list(unvisited))
                    else:
                        # find nearest neighbor with random tie-breaking
                        min_dist = min(distance_matrix[current][c] for c in unvisited)
                        candidates = [c for c in unvisited if distance_matrix[current][c] == min_dist]
                        next_node = random.choice(candidates)
                    route.append(next_node)
                    unvisited.remove(next_node)
                    current = next_node
                route.append(0)
                routes.append(route)
        for i in range(truck_count):
            if len(routes[i]) > 2:
                routes[i] = two_opt(routes[i], max_iter=len(clusters[i])*2)
        lengths = [route_length(r) for r in routes]
        routes, lengths = balance_routes(routes, lengths)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        # perturbation loop
        max_iter = min(200, n * truck_count)
        current_routes = [r[:] for r in routes]
        current_lengths = lengths[:]
        no_improve = 0
        shake_interval = min(30, n)
        for _ in range(max_iter):
            # swap perturbation
            usable = [i for i, r in enumerate(current_routes) if len(r) > 2]
            if len(usable) < 2:
                break
            src = random.choice(usable)
            dst = random.choice([i for i in usable if i != src])
            pos_src = random.randint(1, len(current_routes[src])-2)
            pos_dst = random.randint(1, len(current_routes[dst])-2)
            cust_src = current_routes[src][pos_src]
            cust_dst = current_routes[dst][pos_dst]
            new_src = current_routes[src][:]
            new_dst = current_routes[dst][:]
            new_src[pos_src] = cust_dst
            new_dst[pos_dst] = cust_src
            new_routes = current_routes[:]
            new_routes[src] = new_src
            new_routes[dst] = new_dst
            for i in (src, dst):
                if len(new_routes[i]) > 2:
                    new_routes[i] = two_opt(new_routes[i], max_iter=5)
            new_lengths = [route_length(r) for r in new_routes]
            new_routes, new_lengths = balance_routes(new_routes, new_lengths)
            new_max = max(new_lengths)
            current_routes = new_routes
            current_lengths = new_lengths
            if new_max < best_max:
                best_max = new_max
                best_routes = [r[:] for r in current_routes]
                report_best_vrp(best_routes)
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= shake_interval:
                current_routes, current_lengths = shake(current_routes, current_lengths, alpha=0.05)
                no_improve = 0
        # after perturbation loop, check current
        current_max = max(current_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in current_routes]
            report_best_vrp(best_routes)
    return best_routes