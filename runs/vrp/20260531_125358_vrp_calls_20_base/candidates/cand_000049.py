import random
import numpy as np

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
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
        for i in range(len(route) - 1):
            total += distance_matrix[route[i]][route[i + 1]]
        return total

    def two_opt(route, max_iter=10):
        route = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    if route_length(new_route) < route_length(route):
                        route = new_route
                        improved = True
        return route

    def or_opt(route, max_iter=5):
        route = route[:]
        for _ in range(max_iter):
            improved = False
            for pos in range(1, len(route) - 1):
                cust = route[pos]
                new_route = route[:pos] + route[pos + 1:]
                best_len = route_length(new_route)
                best_pos = -1
                for insert_pos in range(1, len(new_route)):
                    candidate = new_route[:insert_pos] + [cust] + new_route[insert_pos:]
                    l = route_length(candidate)
                    if l < best_len:
                        best_len = l
                        best_pos = insert_pos
                if best_pos != -1:
                    route = new_route[:best_pos] + [cust] + new_route[best_pos:]
                    improved = True
                    break
            if not improved:
                break
        return route

    def balance_and_swap(routes, lengths):
        improved = True
        max_iter = n * truck_count
        bal_it = 0
        no_improve_count = 0
        while improved and bal_it < max_iter and no_improve_count < 5:
            improved = False
            bal_it += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            max_route = routes[max_idx]
            best_reduction = 0
            best_move = None
            # Try moving a customer from max_route to any other route
            for dst_idx in range(truck_count):
                if dst_idx == max_idx:
                    continue
                dst_route = routes[dst_idx]
                for pos in range(1, len(max_route) - 1):
                    cust = max_route[pos]
                    new_max = max_route[:pos] + max_route[pos + 1:]
                    new_max_len = route_length(new_max)
                    best_insert_len = float('inf')
                    best_insert_pos = -1
                    for p in range(1, len(dst_route)):
                        candidate = dst_route[:p] + [cust] + dst_route[p:]
                        l = route_length(candidate)
                        if l < best_insert_len:
                            best_insert_len = l
                            best_insert_pos = p
                    if best_insert_pos == -1:
                        continue
                    new_dst = dst_route[:best_insert_pos] + [cust] + dst_route[best_insert_pos:]
                    new_lengths = lengths[:]
                    new_lengths[max_idx] = new_max_len
                    new_lengths[dst_idx] = route_length(new_dst)
                    new_max_val = max(new_lengths)
                    old_max_val = max(lengths)
                    reduction = old_max_val - new_max_val
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_move = ('move', max_idx, dst_idx, cust, None, new_max, new_dst)
            # Try swapping a customer from max_route with a customer from another route
            for dst_idx in range(truck_count):
                if dst_idx == max_idx:
                    continue
                dst_route = routes[dst_idx]
                for pos_m in range(1, len(max_route) - 1):
                    cust_m = max_route[pos_m]
                    for pos_d in range(1, len(dst_route) - 1):
                        cust_d = dst_route[pos_d]
                        new_max = max_route[:pos_m] + [cust_d] + max_route[pos_m + 1:]
                        new_dst = dst_route[:pos_d] + [cust_m] + dst_route[pos_d + 1:]
                        new_max_len = route_length(new_max)
                        new_dst_len = route_length(new_dst)
                        new_lengths = lengths[:]
                        new_lengths[max_idx] = new_max_len
                        new_lengths[dst_idx] = new_dst_len
                        new_max_val = max(new_lengths)
                        old_max_val = max(lengths)
                        reduction = old_max_val - new_max_val
                        if reduction > best_reduction:
                            best_reduction = reduction
                            best_move = ('swap', max_idx, dst_idx, cust_m, cust_d, new_max, new_dst)
            if best_move is not None and best_reduction > 0:
                move_type, src, dst, cust1, cust2, new_src_route, new_dst_route = best_move
                routes[src] = new_src_route
                routes[dst] = new_dst_route
                lengths[src] = route_length(new_src_route)
                lengths[dst] = route_length(new_dst_route)
                improved = True
                no_improve_count = 0
                report_best_vrp(routes)
            else:
                no_improve_count += 1
        return routes, lengths

    best_routes = None
    best_max = float('inf')
    num_restarts = min(20, max(1, n // 5))
    for restart_idx in range(num_restarts):
        if num_restarts > 1:
            random_prob = 0.3 - 0.25 * (restart_idx / (num_restarts - 1))
        else:
            random_prob = 0.3
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
                routes.append([0, 0])
            else:
                unvisited = set(clusters[i])
                route = [0]
                current = 0
                while unvisited:
                    if random.random() < random_prob:
                        next_node = random.choice(list(unvisited))
                    else:
                        min_dist = min(unvisited, key=lambda x: distance_matrix[current][x])
                        candidates = [x for x in unvisited if distance_matrix[current][x] == distance_matrix[current][min_dist]]
                        next_node = min(candidates)
                    route.append(next_node)
                    unvisited.remove(next_node)
                    current = next_node
                route.append(0)
                routes.append(route)
        for i in range(truck_count):
            if len(routes[i]) > 2:
                routes[i] = two_opt(routes[i], max_iter=len(clusters[i]) * 3)
                routes[i] = or_opt(routes[i], max_iter=5)
        lengths = [route_length(r) for r in routes]
        routes, lengths = balance_and_swap(routes, lengths)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        # Additional improvement loop (no random perturbation)
        imp_iter = n * truck_count
        for _ in range(imp_iter):
            # Intra-route improvement
            for i in range(truck_count):
                if len(routes[i]) > 2:
                    routes[i] = two_opt(routes[i], max_iter=10)
                    routes[i] = or_opt(routes[i], max_iter=3)
            # Inter-route improvement
            routes, lengths = balance_and_swap(routes, lengths)
            new_max = max(lengths)
            if new_max < best_max:
                best_max = new_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
    return best_routes