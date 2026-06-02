def solve_vrp(distance_matrix, truck_count):
    import math
    import random
    import numpy as np
    random.seed(0)
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0, 0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    def compute_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # Greedy construction minimizing max route distance
    routes = [[0, 0] for _ in range(truck_count)]
    dists = [0.0 for _ in range(truck_count)]
    customers = list(range(1, n))
    random.shuffle(customers)
    for cust in customers:
        best_new_max = math.inf
        best_new_total = math.inf
        best_route_idx = None
        best_pos = None
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                insert_cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                new_dist = dists[r_idx] + insert_cost
                other_dists = [dists[i] for i in range(truck_count) if i != r_idx]
                candidate_max = max(other_dists + [new_dist])
                candidate_total = sum(other_dists) + new_dist
                if (candidate_max < best_new_max) or (candidate_max == best_new_max and candidate_total < best_new_total):
                    best_new_max = candidate_max
                    best_new_total = candidate_total
                    best_route_idx = r_idx
                    best_pos = pos
        if best_route_idx is not None:
            routes[best_route_idx].insert(best_pos, cust)
            dists[best_route_idx] = compute_dist(routes[best_route_idx])
    report_best_vrp(routes)

    best_routes = [list(r) for r in routes]
    best_dists = list(dists)
    best_max = max(dists)
    best_total = sum(dists)

    # Intra-route 2-opt
    def apply_2opt(routes, dists):
        for idx in range(len(routes)):
            route = routes[idx]
            if len(route) <= 3:
                continue
            improved = True
            while improved:
                improved = False
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        if k - i == 1:
                            continue
                        old_cost = distance_matrix[route[i-1], route[i]] + distance_matrix[route[k], route[k+1]]
                        new_cost = distance_matrix[route[i-1], route[k]] + distance_matrix[route[i], route[k+1]]
                        if new_cost < old_cost - 1e-12:
                            route[i:k+1] = route[i:k+1][::-1]
                            improved = True
                            new_dist = compute_dist(route)
                            dists[idx] = new_dist
                            yield (routes, dists)
                            break
                    if improved:
                        break
        return

    # Inter-route relocate and swap (best improvement)
    def improve(routes, dists, max_dist, total_dist):
        max_iter = n * truck_count
        for iteration in range(max_iter):
            order = sorted(range(len(routes)), key=lambda idx: dists[idx], reverse=True)
            improved = False
            for i_route in order:
                if improved:
                    break
                best_new_max = max_dist
                best_new_total = total_dist
                best_move = None
                route_i = routes[i_route]
                # Relocate
                for pos in range(1, len(route_i) - 1):
                    customer = route_i[pos]
                    prev = route_i[pos-1]
                    nxt = route_i[pos+1]
                    removal_saving = distance_matrix[prev, customer] + distance_matrix[customer, nxt] - distance_matrix[prev, nxt]
                    new_dist_i = dists[i_route] - removal_saving
                    if new_dist_i < 0:
                        continue
                    for j_route in range(len(routes)):
                        if j_route == i_route:
                            continue
                        route_j = routes[j_route]
                        best_insert_cost = math.inf
                        best_insert_pos = None
                        for k in range(1, len(route_j)):
                            pred = route_j[k-1]
                            succ = route_j[k]
                            insert_cost = distance_matrix[pred, customer] + distance_matrix[customer, succ] - distance_matrix[pred, succ]
                            if insert_cost < best_insert_cost:
                                best_insert_cost = insert_cost
                                best_insert_pos = k
                        if best_insert_pos is None:
                            continue
                        new_dist_j = dists[j_route] + best_insert_cost
                        other_dists = [dists[idx] for idx in range(len(routes)) if idx not in (i_route, j_route)]
                        combined = other_dists + [new_dist_i, new_dist_j]
                        candidate_max = max(combined)
                        candidate_total = total_dist - removal_saving + best_insert_cost
                        if (candidate_max < best_new_max - 1e-12) or (abs(candidate_max - best_new_max) < 1e-12 and candidate_total < best_new_total - 1e-12):
                            best_new_max = candidate_max
                            best_new_total = candidate_total
                            best_move = ('relocate', i_route, pos, j_route, best_insert_pos, new_dist_i, new_dist_j)
                # Swap
                for j_route in range(len(routes)):
                    if j_route == i_route:
                        continue
                    route_j = routes[j_route]
                    for pos_i in range(1, len(route_i) - 1):
                        cust_i = route_i[pos_i]
                        for pos_j in range(1, len(route_j) - 1):
                            cust_j = route_j[pos_j]
                            prev_i = route_i[pos_i-1]
                            next_i = route_i[pos_i+1]
                            saving_i = distance_matrix[prev_i, cust_i] + distance_matrix[cust_i, next_i] - distance_matrix[prev_i, next_i]
                            add_i = distance_matrix[prev_i, cust_j] + distance_matrix[cust_j, next_i] - distance_matrix[prev_i, next_i]
                            new_dist_i = dists[i_route] - saving_i + add_i
                            prev_j = route_j[pos_j-1]
                            next_j = route_j[pos_j+1]
                            saving_j = distance_matrix[prev_j, cust_j] + distance_matrix[cust_j, next_j] - distance_matrix[prev_j, next_j]
                            add_j = distance_matrix[prev_j, cust_i] + distance_matrix[cust_i, next_j] - distance_matrix[prev_j, next_j]
                            new_dist_j = dists[j_route] - saving_j + add_j
                            other_dists = [dists[idx] for idx in range(len(routes)) if idx not in (i_route, j_route)]
                            combined = other_dists + [new_dist_i, new_dist_j]
                            candidate_max = max(combined)
                            candidate_total = total_dist - saving_i + add_i - saving_j + add_j
                            if (candidate_max < best_new_max - 1e-12) or (abs(candidate_max - best_new_max) < 1e-12 and candidate_total < best_new_total - 1e-12):
                                best_new_max = candidate_max
                                best_new_total = candidate_total
                                best_move = ('swap', i_route, pos_i, j_route, pos_j, new_dist_i, new_dist_j)
                if best_move is not None:
                    if best_move[0] == 'relocate':
                        _, i_route, pos, j_route, insert_pos, new_dist_i, new_dist_j = best_move
                        customer = routes[i_route].pop(pos)
                        dists[i_route] = new_dist_i
                        routes[j_route].insert(insert_pos, customer)
                        dists[j_route] = new_dist_j
                    else:
                        _, i_route, pos_i, j_route, pos_j, new_dist_i, new_dist_j = best_move
                        cust_i = routes[i_route][pos_i]
                        cust_j = routes[j_route][pos_j]
                        routes[i_route][pos_i] = cust_j
                        routes[j_route][pos_j] = cust_i
                        dists[i_route] = new_dist_i
                        dists[j_route] = new_dist_j
                    total_dist = best_new_total
                    max_dist = best_new_max
                    yield (routes, dists, max_dist, total_dist)
                    improved = True
                    break
            if not improved:
                break
        return

    # Initial improvement
    for _ in apply_2opt(routes, dists):
        report_best_vrp(routes)
    for _ in improve(routes, dists, max(dists), sum(dists)):
        report_best_vrp(routes)
        if max(dists) < best_max - 1e-12 or (abs(max(dists)-best_max) < 1e-12 and sum(dists) < best_total - 1e-12):
            best_max = max(dists)
            best_total = sum(dists)
            best_routes = [list(r) for r in routes]
            best_dists = list(dists)

    # Main ILS loop
    max_restarts = 10
    for restart in range(max_restarts):
        # Ruin-and-recreate perturbation
        routes = [list(r) for r in best_routes]
        dists = list(best_dists)
        # Remove a random fraction of customers (20%)
        num_removals = max(1, int(0.2 * (n - 1)))
        all_customers = [i for i in range(1, n)]
        random.shuffle(all_customers)
        to_remove = set(all_customers[:num_removals])
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            new_route = [0]
            for node in route[1:-1]:
                if node not in to_remove:
                    new_route.append(node)
            new_route.append(0)
            routes[r_idx] = new_route
            dists[r_idx] = compute_dist(new_route)
        # Reinsert removed customers greedily minimizing max
        for cust in to_remove:
            best_new_max = math.inf
            best_new_total = math.inf
            best_route_idx = None
            best_pos = None
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    insert_cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_dist = dists[r_idx] + insert_cost
                    other_dists = [dists[i] for i in range(truck_count) if i != r_idx]
                    candidate_max = max(other_dists + [new_dist])
                    candidate_total = sum(other_dists) + new_dist
                    if (candidate_max < best_new_max) or (candidate_max == best_new_max and candidate_total < best_new_total):
                        best_new_max = candidate_max
                        best_new_total = candidate_total
                        best_route_idx = r_idx
                        best_pos = pos
            if best_route_idx is not None:
                routes[best_route_idx].insert(best_pos, cust)
                dists[best_route_idx] = compute_dist(routes[best_route_idx])
        report_best_vrp(routes)
        # Improvement
        for _ in apply_2opt(routes, dists):
            report_best_vrp(routes)
        for _ in improve(routes, dists, max(dists), sum(dists)):
            report_best_vrp(routes)
            if max(dists) < best_max - 1e-12 or (abs(max(dists)-best_max) < 1e-12 and sum(dists) < best_total - 1e-12):
                best_max = max(dists)
                best_total = sum(dists)
                best_routes = [list(r) for r in routes]
                best_dists = list(dists)

    report_best_vrp(best_routes)
    return best_routes