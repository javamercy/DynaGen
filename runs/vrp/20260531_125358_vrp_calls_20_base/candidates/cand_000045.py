def solve_vrp(distance_matrix, truck_count):
    import random
    import numpy as np
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def report_best_vrp(routes):
        pass

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

    def balance_routes(routes, lengths):
        improved = True
        max_bal_iter = n * truck_count
        bal_it = 0
        no_improve_count = 0
        while improved and bal_it < max_bal_iter and no_improve_count < 2:
            improved = False
            bal_it += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            # Evaluate all moves from max route to every other route
            best_move = None
            best_reduction = 0
            max_route = routes[max_idx]
            if len(max_route) <= 2:
                break
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_length(new_max_route)
                for target_idx in range(truck_count):
                    if target_idx == max_idx:
                        continue
                    target_route = routes[target_idx]
                    for ins_pos in range(1, len(target_route)):
                        new_target_route = target_route[:ins_pos] + [cust] + target_route[ins_pos:]
                        new_target_len = route_length(new_target_route)
                        other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, target_idx)]
                        new_max_global = max(new_max_len, new_target_len, max(other_lengths) if other_lengths else 0)
                        old_max_global = max(lengths)
                        reduction = old_max_global - new_max_global
                        if reduction > best_reduction:
                            best_reduction = reduction
                            best_move = (max_idx, target_idx, pos, ins_pos, cust)
            if best_move is not None and best_reduction > 0:
                src_idx, dst_idx, src_pos, dst_pos, cust = best_move
                src_route = routes[src_idx]
                dst_route = routes[dst_idx]
                new_src = src_route[:src_pos] + src_route[src_pos+1:]
                new_dst = dst_route[:dst_pos] + [cust] + dst_route[dst_pos:]
                routes[src_idx] = new_src
                routes[dst_idx] = new_dst
                lengths[src_idx] = route_length(new_src)
                lengths[dst_idx] = route_length(new_dst)
                improved = True
                no_improve_count = 0
                report_best_vrp(routes)
            else:
                no_improve_count += 1
        return routes, lengths

    best_routes = None
    best_max = float('inf')
    num_restarts = min(20, max(1, n//5))
    for restart_idx in range(num_restarts):
        if num_restarts > 1:
            random_prob = 0.4 - 0.3 * (restart_idx / (num_restarts - 1))
        else:
            random_prob = 0.4
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
                    if random.random() < random_prob:
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
        routes, lengths = balance_routes(routes, lengths)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        max_perturb = min(50, n * truck_count)
        current_routes = [r[:] for r in routes]
        current_lengths = lengths[:]
        for _ in range(max_perturb):
            usable = [i for i, r in enumerate(current_routes) if len(r) > 2]
            if len(usable) < 2:
                break
            src = random.choice(usable)
            pos = random.randint(1, len(current_routes[src])-2)
            cust = current_routes[src][pos]
            dst = random.choice([i for i in range(truck_count) if i != src])
            new_src = [node for node in current_routes[src] if node != cust]
            dst_route = current_routes[dst]
            if len(dst_route) == 0:
                dst_route = [0,0]
            ins_pos = random.randint(1, len(dst_route)-1)
            new_dst = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
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
    if best_routes is None:
        best_routes = [[0,0] for _ in range(truck_count)]
        idx = 0
        for c in customers:
            if idx < truck_count:
                best_routes[idx] = [0, c, 0]
                idx += 1
            else:
                best_routes[-1].insert(-1, c)
    return best_routes