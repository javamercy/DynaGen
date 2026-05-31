import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def two_opt(route, max_iter=5):
        route = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
        return route

    def randomized_cheapest_insertion(epsilon=0.1):
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        # random perturbation
                        inc_perturbed = inc * (1 + random.uniform(0, epsilon))
                        if inc_perturbed < best_inc:
                            best_inc = inc_perturbed
                            best_cust = cust
                            best_route_idx = r_idx
                            best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes

    def balance_routes_simple(routes, lengths):
        improved = True
        max_iter_balance = n
        it = 0
        while improved and it < max_iter_balance:
            improved = False
            it += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == 0:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_pos = -1
            best_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max_route)
                min_route = routes[min_idx]
                best_ins_len = float('inf')
                best_ins_pos = -1
                for p in range(1, len(min_route)):
                    new_min_route = min_route[:p] + [cust] + min_route[p:]
                    l = route_distance(new_min_route)
                    if l < best_ins_len:
                        best_ins_len = l
                        best_ins_pos = p
                new_min_route = min_route[:best_ins_pos] + [cust] + min_route[best_ins_pos:]
                new_min_len = route_distance(new_min_route)
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                old_max = lengths[max_idx]
                new_max = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                reduction = old_max - new_max
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_cust = cust
                    best_pos = best_ins_pos
            if best_cust is not None and best_reduction > 0:
                new_max_route = [node for node in max_route if node != best_cust]
                min_route = routes[min_idx]
                new_min_route = min_route[:best_pos] + [best_cust] + min_route[best_pos:]
                routes[max_idx] = new_max_route
                routes[min_idx] = new_min_route
                lengths[max_idx] = route_distance(new_max_route)
                lengths[min_idx] = route_distance(new_min_route)
                improved = True
            else:
                break
        return routes, lengths

    def ruin_recreate(routes, lengths, fraction):
        n_cust = n - 1
        num_remove = max(1, int(n_cust * fraction))
        custs = list(range(1, n))
        random.shuffle(custs)
        to_remove = custs[:num_remove]
        new_routes = [[0, 0] for _ in range(truck_count)]
        for r_idx, route in enumerate(routes):
            new_routes[r_idx] = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
        unvisited = set(to_remove)
        while unvisited:
            best_cust = None
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                for r_idx, route in enumerate(new_routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        if inc < best_inc:
                            best_inc = inc
                            best_cust = cust
                            best_route_idx = r_idx
                            best_pos = pos
            new_routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        for r_idx in range(truck_count):
            if len(new_routes[r_idx]) > 2:
                new_routes[r_idx] = two_opt(new_routes[r_idx], max_iter=5)
        new_lengths = [route_distance(r) for r in new_routes]
        return new_routes, new_lengths

    best_routes = None
    best_max = float('inf')
    num_restarts = max(1, min(5, n//10))
    for restart in range(num_restarts):
        routes = randomized_cheapest_insertion(epsilon=0.1)
        lengths = [route_distance(r) for r in routes]
        routes, lengths = balance_routes_simple(routes, lengths)
        current_max = max(lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

        max_iter = n * truck_count
        ruin_fraction = 0.15
        no_improve_count = 0
        for iteration in range(max_iter):
            best_move = None
            best_new_max = current_max
            best_total = sum(lengths)

            # Inter-route relocate
            for cust in range(1, n):
                src_idx = None
                src_pos = None
                for r_idx, route in enumerate(routes):
                    if cust in route:
                        src_idx = r_idx
                        src_pos = route.index(cust)
                        break
                if src_idx is None:
                    continue
                new_src_route = routes[src_idx][:src_pos] + routes[src_idx][src_pos+1:]
                src_len = route_distance(new_src_route)
                for dst_idx in range(truck_count):
                    if dst_idx == src_idx:
                        continue
                    dst_route = routes[dst_idx]
                    if len(dst_route) <= 2:
                        continue
                    for ins_pos in range(1, len(dst_route)):
                        new_dst_route = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                        new_lengths = lengths[:]
                        new_lengths[src_idx] = src_len
                        new_lengths[dst_idx] = route_distance(new_dst_route)
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if (new_max < best_new_max or
                            (new_max == best_new_max and new_total < best_total) or
                            (new_max == best_new_max and new_total == best_total and (src_idx < dst_idx or (src_idx == dst_idx and ins_pos < 0)))):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('relocate', cust, src_idx, src_pos, dst_idx, ins_pos, new_src_route, new_dst_route)

            # Inter-route swap
            for cust1 in range(1, n):
                src_idx = None
                src_pos = None
                for r_idx, route in enumerate(routes):
                    if cust1 in route:
                        src_idx = r_idx
                        src_pos = route.index(cust1)
                        break
                if src_idx is None:
                    continue
                for cust2 in range(1, n):
                    if cust2 == cust1:
                        continue
                    dst_idx = None
                    dst_pos = None
                    for r_idx, route in enumerate(routes):
                        if cust2 in route:
                            dst_idx = r_idx
                            dst_pos = route.index(cust2)
                            break
                    if dst_idx is None or dst_idx == src_idx:
                        continue
                    new_src_route = routes[src_idx][:]
                    new_src_route[src_pos] = cust2
                    new_dst_route = routes[dst_idx][:]
                    new_dst_route[dst_pos] = cust1
                    new_lengths = lengths[:]
                    new_lengths[src_idx] = route_distance(new_src_route)
                    new_lengths[dst_idx] = route_distance(new_dst_route)
                    new_max = max(new_lengths)
                    new_total = sum(new_lengths)
                    if (new_max < best_new_max or
                        (new_max == best_new_max and new_total < best_total) or
                        (new_max == best_new_max and new_total == best_total and (src_idx < dst_idx or (src_idx == dst_idx and src_pos < dst_pos)))):
                        best_new_max = new_max
                        best_total = new_total
                        best_move = ('swap', cust1, cust2, src_idx, src_pos, dst_idx, dst_pos, new_src_route, new_dst_route)

            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_len = route_distance(new_route)
                        if new_len >= lengths[r_idx]:
                            continue
                        new_lengths = lengths[:]
                        new_lengths[r_idx] = new_len
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if (new_max < best_new_max or
                            (new_max == best_new_max and new_total < best_total) or
                            (new_max == best_new_max and new_total == best_total and r_idx < 0)):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('2opt', r_idx, i, j, new_route)

            if best_move is not None and best_new_max < current_max:
                if best_move[0] == 'relocate':
                    routes[best_move[2]] = best_move[6]
                    routes[best_move[4]] = best_move[7]
                elif best_move[0] == 'swap':
                    routes[best_move[2]] = best_move[7]
                    routes[best_move[5]] = best_move[8]
                elif best_move[0] == '2opt':
                    routes[best_move[1]] = best_move[4]
                lengths = [route_distance(r) for r in routes]
                current_max = max(lengths)
                no_improve_count = 0
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
            else:
                no_improve_count += 1
                # Adapt ruin fraction or shake
                if no_improve_count >= 10:
                    # shake: apply ruin with larger fraction
                    routes, lengths = ruin_recreate(routes, lengths, fraction=0.3)
                    current_max = max(lengths)
                    no_improve_count = 0
                    ruin_fraction = 0.15
                else:
                    # increase ruin fraction gradually
                    ruin_fraction = min(0.3, 0.15 + 0.05 * (iteration % 10))
                    routes, lengths = ruin_recreate(routes, lengths, fraction=ruin_fraction)
                    current_max = max(lengths)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)

    return best_routes