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
        return route
    best_routes = None
    best_max = float('inf')
    num_restarts = min(10, max(1, n//10))
    for _ in range(num_restarts):
        seeds = random.sample(customers, min(truck_count, len(customers)))
        clusters = [[] for _ in range(truck_count)]
        for i, s in enumerate(seeds):
            clusters[i].append(s)
        remaining = [c for c in customers if c not in seeds]
        for cust in remaining:
            best_cluster = 0
            best_dist = distance_matrix[cust][seeds[0]]
            for i in range(1, truck_count):
                d = distance_matrix[cust][seeds[i]]
                if d < best_dist:
                    best_dist = d
                    best_cluster = i
                elif d == best_dist and i < best_cluster:
                    best_cluster = i
            clusters[best_cluster].append(cust)
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
                        next_node = min(unvisited, key=lambda x: distance_matrix[current][x])
                    route.append(next_node)
                    unvisited.remove(next_node)
                    current = next_node
                route.append(0)
                routes.append(route)
        for i in range(truck_count):
            if len(routes[i]) > 2:
                routes[i] = two_opt(routes[i], max_iter=len(clusters[i])*2)
        lengths = [route_length(r) for r in routes]
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        max_perturb = min(100, n * truck_count)
        current_routes = [r[:] for r in routes]
        current_lengths = lengths[:]
        for _ in range(max_perturb):
            usable = [i for i, r in enumerate(current_routes) if len(r) > 2]
            if len(usable) < 2:
                break
            r1 = random.choice(usable)
            r2 = random.choice([i for i in usable if i != r1])
            pos1 = random.randint(1, len(current_routes[r1])-2)
            pos2 = random.randint(1, len(current_routes[r2])-2)
            cust1 = current_routes[r1][pos1]
            cust2 = current_routes[r2][pos2]
            new_r1 = current_routes[r1][:pos1] + current_routes[r1][pos1+1:]
            new_r2 = current_routes[r2][:pos2] + current_routes[r2][pos2+1:]
            new_r1 = new_r1[:pos1] + [cust2] + new_r1[pos1:]
            new_r2 = new_r2[:pos2] + [cust1] + new_r2[pos2:]
            new_routes = current_routes[:]
            new_routes[r1] = new_r1
            new_routes[r2] = new_r2
            for i in (r1, r2):
                if len(new_routes[i]) > 2:
                    new_routes[i] = two_opt(new_routes[i], max_iter=5)
            new_lengths = [route_length(r) for r in new_routes]
            new_max = max(new_lengths)
            current_routes = new_routes
            current_lengths = new_lengths
            if new_max < best_max:
                best_max = new_max
                best_routes = [r[:] for r in current_routes]
                report_best_vrp(best_routes)
    return best_routes