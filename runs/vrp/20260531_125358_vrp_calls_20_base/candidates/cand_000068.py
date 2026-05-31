def solve_vrp(distance_matrix, truck_count):
    import random
    import math
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = []
        for i in range(truck_count):
            if i < len(customers):
                routes.append([0, customers[i], 0])
            else:
                routes.append([0, 0])
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
    num_restarts = 20
    for restart in range(num_restarts):
        random.seed(restart)
        # Farthest-first clustering with random tie-breaking
        seeds = []
        farthest_cust = max(customers, key=lambda x: (distance_matrix[0][x], random.random()))
        seeds.append(farthest_cust)
        for _ in range(min(truck_count, len(customers)) - 1):
            min_dist_to_seeds = {}
            for c in customers:
                if c not in seeds:
                    min_dist = min(distance_matrix[c][s] for s in seeds)
                    min_dist_to_seeds[c] = (min_dist, random.random())
            candidate = max(min_dist_to_seeds, key=lambda c: min_dist_to_seeds[c])
            seeds.append(candidate)
        # Assign customers to nearest seed with random tie-breaking
        clusters = [[] for _ in range(truck_count)]
        for i, s in enumerate(seeds):
            clusters[i].append(s)
        remaining = [c for c in customers if c not in seeds]
        for cust in remaining:
            best_cluster = 0
            best_dist = distance_matrix[cust][seeds[0]]
            ties = [0]
            for i in range(1, truck_count):
                d = distance_matrix[cust][seeds[i]]
                if d < best_dist:
                    best_dist = d
                    best_cluster = i
                    ties = [i]
                elif d == best_dist:
                    ties.append(i)
            best_cluster = random.choice(ties)
            clusters[best_cluster].append(cust)
        # Build routes with nearest neighbor random tie-breaking
        routes = []
        for i in range(truck_count):
            if not clusters[i]:
                routes.append([0, 0])
            else:
                unvisited = set(clusters[i])
                route = [0]
                current = 0
                while unvisited:
                    candidates = [x for x in unvisited if distance_matrix[current][x] == min(distance_matrix[current][x] for x in unvisited)]
                    next_node = random.choice(candidates)
                    route.append(next_node)
                    unvisited.remove(next_node)
                    current = next_node
                route.append(0)
                routes.append(route)
        # 2-opt
        for i in range(truck_count):
            if len(routes[i]) > 2:
                routes[i] = two_opt(routes[i], max_iter=len(clusters[i]) * 2)
        lengths = [route_length(r) for r in routes]
        routes, lengths = balance_routes(routes, lengths)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        # Simulated annealing perturbation
        temperature = max(lengths) * 0.1
        cooling = 0.99
        max_perturb = 200
        current_routes = [r[:] for r in routes]
        current_lengths = lengths[:]
        for _ in range(max_perturb):
            # Choose perturbation type: relocate or swap
            if random.random() < 0.5:
                # Random relocate
                usable = [i for i, r in enumerate(current_routes) if len(r) > 2]
                if len(usable) < 2:
                    break
                src = random.choice(usable)
                pos = random.randint(1, len(current_routes[src]) - 2)
                cust = current_routes[src][pos]
                dst = random.choice([i for i in range(truck_count) if i != src])
                new_src = [node for node in current_routes[src] if node != cust]
                dst_route = current_routes[dst]
                if len(dst_route) == 0:
                    dst_route = [0, 0]
                ins_pos = random.randint(1, len(dst_route) - 1)
                new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                new_routes = current_routes[:]
                new_routes[src] = new_src
                new_routes[dst] = new_dst
            else:
                # Swap between two routes
                usable = [i for i, r in enumerate(current_routes) if len(r) > 2]
                if len(usable) < 2:
                    break
                r1, r2 = random.sample(usable, 2)
                if len(current_routes[r1]) <= 2 or len(current_routes[r2]) <= 2:
                    continue
                pos1 = random.randint(1, len(current_routes[r1]) - 2)
                pos2 = random.randint(1, len(current_routes[r2]) - 2)
                cust1 = current_routes[r1][pos1]
                cust2 = current_routes[r2][pos2]
                new_r1 = current_routes[r1][:pos1] + [cust2] + current_routes[r1][pos1+1:]
                new_r2 = current_routes[r2][:pos2] + [cust1] + current_routes[r2][pos2+1:]
                new_routes = current_routes[:]
                new_routes[r1] = new_r1
                new_routes[r2] = new_r2
            for i in (r1 if 'r1' in dir() else src, r2 if 'r2' in dir() else dst):
                if len(new_routes[i]) > 2:
                    new_routes[i] = two_opt(new_routes[i], max_iter=5)
            new_lengths = [route_length(r) for r in new_routes]
            new_routes, new_lengths = balance_routes(new_routes, new_lengths)
            new_max = max(new_lengths)
            delta = new_max - current_max
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_routes = new_routes
                current_lengths = new_lengths
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [r[:] for r in current_routes]
                    report_best_vrp(best_routes)
            temperature *= cooling
    return best_routes