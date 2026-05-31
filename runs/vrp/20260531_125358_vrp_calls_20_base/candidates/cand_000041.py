def solve_vrp(distance_matrix, truck_count):
    import random, math, copy
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

    def compute_lengths(routes):
        return [route_length(r) for r in routes]

    def shake(routes, lengths, k=2):
        all_customers = []
        for i, r in enumerate(routes):
            for c in r[1:-1]:
                all_customers.append((i, c))
        selected = random.sample(all_customers, min(k, len(all_customers)))
        for src, cust in selected:
            routes[src] = [0] + [c for c in routes[src][1:-1] if c != cust] + [0]
        for src, cust in selected:
            possible = [i for i in range(truck_count) if i != src]
            if not possible:
                target = src
            else:
                target = random.choice(possible)
            if len(routes[target]) <= 2:
                routes[target] = [0, cust, 0]
            else:
                pos = random.randint(1, len(routes[target])-2)
                routes[target] = routes[target][:pos] + [cust] + routes[target][pos:]
        return routes

    best_routes = None
    best_max = float('inf')
    num_restarts = min(20, max(1, n//5))
    for _ in range(num_restarts):
        seeds = random.sample(customers, min(truck_count, len(customers)))
        clusters = [[] for _ in range(truck_count)]
        for i, s in enumerate(seeds):
            clusters[i].append(s)
        remaining = [c for c in customers if c not in seeds]
        for cust in remaining:
            if random.random() < 0.3:
                truck = random.randint(0, truck_count-1)
            else:
                best_truck = 0
                best_dist = distance_matrix[cust][seeds[0]]
                for i in range(1, truck_count):
                    d = distance_matrix[cust][seeds[i]]
                    if d < best_dist:
                        best_dist = d
                        best_truck = i
                truck = best_truck
            clusters[truck].append(cust)
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
        lengths = compute_lengths(routes)
        improved = True
        while improved:
            improved = False
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_pos = None
            best_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_length(new_max)
                min_route = routes[min_idx]
                best_ins_len = float('inf')
                best_ins_pos = -1
                for p in range(1, len(min_route)):
                    candidate = min_route[:p] + [cust] + min_route[p:]
                    l = route_length(candidate)
                    if l < best_ins_len:
                        best_ins_len = l
                        best_ins_pos = p
                new_min = min_route[:best_ins_pos] + [cust] + min_route[best_ins_pos:]
                new_min_len = route_length(new_min)
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_cust = (cust, best_ins_pos)
            if best_cust is not None:
                cust, ins_pos = best_cust
                new_max = [node for node in max_route if node != cust]
                min_route = routes[min_idx]
                new_min = min_route[:ins_pos] + [cust] + min_route[ins_pos:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_length(new_max)
                lengths[min_idx] = route_length(new_min)
                improved = True
                report_best_vrp(routes)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        max_perturb = min(200, n * truck_count * 2)
        current_routes = [r[:] for r in routes]
        current_lengths = lengths[:]
        T = 0.1 * current_max
        for step in range(max_perturb):
            if random.random() < 0.5:
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
            else:
                usable = [i for i, r in enumerate(current_routes) if len(r) > 2]
                if len(usable) == 0:
                    break
                src = random.choice(usable)
                pos = random.randint(1, len(current_routes[src])-2)
                cust = current_routes[src][pos]
                new_src = current_routes[src][:pos] + current_routes[src][pos+1:]
                target = random.choice([i for i in range(truck_count) if i != src or len(current_routes[i]) <= 2])
                if len(current_routes[target]) <= 2:
                    new_target = [0, cust, 0]
                else:
                    ins_pos = random.randint(1, len(current_routes[target])-2)
                    new_target = current_routes[target][:ins_pos] + [cust] + current_routes[target][ins_pos:]
                new_routes = current_routes[:]
                new_routes[src] = new_src
                new_routes[target] = new_target
            for idx in range(truck_count):
                if len(new_routes[idx]) > 2:
                    new_routes[idx] = two_opt(new_routes[idx], max_iter=5)
            new_lengths = compute_lengths(new_routes)
            new_max = max(new_lengths)
            delta = new_max - current_max
            if delta < 0 or random.random() < math.exp(-delta / (T + 1e-10)):
                current_routes = new_routes
                current_lengths = new_lengths
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [r[:] for r in current_routes]
                    report_best_vrp(best_routes)
            T *= 0.99
            if random.random() < 0.1 and step % 10 == 0:
                current_routes = shake(current_routes, current_lengths, k=3)
                for idx in range(truck_count):
                    if len(current_routes[idx]) > 2:
                        current_routes[idx] = two_opt(current_routes[idx], max_iter=5)
                current_lengths = compute_lengths(current_routes)
                current_max = max(current_lengths)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in current_routes]
                    report_best_vrp(best_routes)
    return best_routes