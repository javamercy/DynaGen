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
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total

    def two_opt(route, max_iter=None):
        if max_iter is None:
            max_iter = len(route) * 2
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
        max_iter = n * truck_count  # bound
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_overall_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = [node for node in max_route if node != cust]
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
                new_min_len = best_insertion_len
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_overall_reduction:
                    best_overall_reduction = reduction
                    best_cust = (cust, best_pos)
            if best_cust is not None and best_overall_reduction > 0:
                cust, best_insert_pos = best_cust
                new_max = [node for node in max_route if node != cust]
                min_route = routes[min_idx]
                new_min = min_route[:best_insert_pos] + [cust] + min_route[best_insert_pos:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_length(new_max)
                lengths[min_idx] = route_length(new_min)
                improved = True
        return routes, lengths

    def construct_deterministic():
        # Farthest-first seed selection
        seeds = []
        first_seed = max(customers, key=lambda x: distance_matrix[0][x])
        seeds.append(first_seed)
        for _ in range(truck_count - 1):
            remaining = [c for c in customers if c not in seeds]
            if not remaining:
                break
            next_seed = max(remaining, key=lambda x: min(distance_matrix[x][s] for s in seeds))
            seeds.append(next_seed)
        # Assign customers to nearest seed
        clusters = [[] for _ in range(truck_count)]
        for i, s in enumerate(seeds):
            clusters[i].append(s)
        remaining_cust = [c for c in customers if c not in seeds]
        for cust in remaining_cust:
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
        # Nearest neighbor construction (deterministic)
        routes = []
        for i in range(truck_count):
            if not clusters[i]:
                routes.append([0, 0])
            else:
                unvisited = sorted(clusters[i])  # deterministic order
                route = [0]
                current = 0
                while unvisited:
                    next_node = min(unvisited, key=lambda x: distance_matrix[current][x])
                    route.append(next_node)
                    unvisited.remove(next_node)
                    current = next_node
                route.append(0)
                routes.append(route)
        return routes

    best_routes = None
    best_max = float('inf')
    num_restarts = 3  # reduce from 10, focus more on local search
    for restart in range(num_restarts):
        # Each restart uses different deterministic construction? Actually same because deterministic.
        # To get variation, we can perturb the seeds order. Instead, we'll use same construction but then apply perturbations.
        routes = construct_deterministic()
        # Initial 2-opt
        for i in range(truck_count):
            if len(routes[i]) > 2:
                routes[i] = two_opt(routes[i], max_iter=len(routes[i])*2)
        lengths = [route_length(r) for r in routes]
        # Initial balancing
        routes, lengths = balance_routes(routes, lengths)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        # Intensification local search: best relocate from longest route
        max_iter = n * truck_count * 2
        it = 0
        improved = True
        while improved and it < max_iter:
            improved = False
            it += 1
            longest_idx = max(range(truck_count), key=lambda i: lengths[i])
            longest_route = routes[longest_idx]
            if len(longest_route) <= 2:
                break
            best_move = None
            best_new_max = current_max
            for pos in range(1, len(longest_route)-1):
                cust = longest_route[pos]
                for other_idx in range(truck_count):
                    if other_idx == longest_idx:
                        continue
                    other_route = routes[other_idx]
                    # Try all insertion positions
                    for ins_pos in range(1, len(other_route)):
                        # compute new lengths without full 2-opt
                        new_long_len = route_length([node for node in longest_route if node != cust])
                        new_other_len = route_length(other_route[:ins_pos] + [cust] + other_route[ins_pos:])
                        max_among_others = max(
                            [lengths[k] for k in range(truck_count) if k not in (longest_idx, other_idx)] +
                            [new_long_len, new_other_len]
                        )
                        if max_among_others < best_new_max:
                            best_new_max = max_among_others
                            best_move = (cust, other_idx, ins_pos)
            if best_move is not None:
                cust, other_idx, ins_pos = best_move
                # Apply move
                routes[longest_idx] = [node for node in routes[longest_idx] if node != cust]
                routes[other_idx] = routes[other_idx][:ins_pos] + [cust] + routes[other_idx][ins_pos:]
                # Run 2-opt on affected routes
                for idx in (longest_idx, other_idx):
                    if len(routes[idx]) > 2:
                        routes[idx] = two_opt(routes[idx], max_iter=len(routes[idx])*2)
                lengths = [route_length(r) for r in routes]
                # Balancing
                routes, lengths = balance_routes(routes, lengths)
                current_max = max(lengths)
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
            else:
                # No improving relocate found, try 2-opt on all routes then balance
                for i in range(truck_count):
                    if len(routes[i]) > 2:
                        routes[i] = two_opt(routes[i], max_iter=len(routes[i])*2)
                lengths = [route_length(r) for r in routes]
                routes, lengths = balance_routes(routes, lengths)
                new_max = max(lengths)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
    return best_routes