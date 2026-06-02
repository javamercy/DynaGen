def solve_vrp(distance_matrix, truck_count):
    import math
    import numpy as np
    import random
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0, 0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    best_routes = None
    best_max_dist = math.inf
    best_total_dist = math.inf

    # Construction: each customer as a single route
    routes = [[0, i, 0] for i in range(1, n)]
    dists = [2 * distance_matrix[0, i] for i in range(1, n)]
    current_max = max(dists) if dists else 0.0

    # Greedy merge until we have truck_count routes
    while len(routes) > truck_count:
        best_new_max = math.inf
        best_new_dist = math.inf
        best_pair = None
        best_merged_route = None
        best_merged_dist = None
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue
                r_i = routes[i]
                r_j = routes[j]
                # Try both orientations
                last_i = r_i[-2]
                first_j = r_j[1]
                dist_ij = dists[i] + dists[j] - distance_matrix[last_i, 0] - distance_matrix[0, first_j] + distance_matrix[last_i, first_j]
                last_j = r_j[-2]
                first_i = r_i[1]
                dist_ji = dists[i] + dists[j] - distance_matrix[last_j, 0] - distance_matrix[0, first_i] + distance_matrix[last_j, first_i]
                if dist_ij <= dist_ji:
                    new_dist = dist_ij
                    merged = r_i[:-1] + r_j[1:]
                else:
                    new_dist = dist_ji
                    merged = r_j[:-1] + r_i[1:]
                new_max = max(current_max, new_dist)
                if (new_max < best_new_max) or (new_max == best_new_max and new_dist < best_new_dist):
                    best_new_max = new_max
                    best_new_dist = new_dist
                    best_pair = (i, j)
                    best_merged_route = merged
                    best_merged_dist = new_dist
        if best_pair is None:
            break
        i, j = best_pair
        routes[i] = best_merged_route
        dists[i] = best_merged_dist
        current_max = best_new_max
        del routes[j]
        del dists[j]

    # Add empty routes if needed
    while len(routes) < truck_count:
        routes.append([0, 0])
        dists.append(0.0)

    total_dist = sum(dists)
    max_dist = max(dists)
    report_best_vrp(routes)

    # Local search + restart loop
    for restart in range(5):
        # Intra-route 2-opt
        for idx in range(len(routes)):
            route = routes[idx]
            if len(route) > 3:
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
                                dists[idx] = compute_route_distance(route)
                                total_dist = sum(dists)
                                max_dist = max(dists)
                                report_best_vrp(routes)
                                break
                        if improved:
                            break

        # Iterative improvement: best-improvement relocate and swap over all routes
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
                    report_best_vrp(routes)
                    improved = True
                    break
            if not improved:
                break

        if restart < 4:
            # Perturbation: randomly relocate 10% customers to new routes
            customers = []
            for idx, route in enumerate(routes):
                for pos in range(1, len(route)-1):
                    customers.append((idx, pos, route[pos]))
            num_to_move = max(1, len(customers) // 10)
            random.shuffle(customers)
            for _ in range(num_to_move):
                idx, pos, cust = customers.pop()
                routes[idx].pop(pos)
                # random target route
                target = random.randrange(len(routes))
                insert_pos = random.randrange(1, len(routes[target]))
                routes[target].insert(insert_pos, cust)
            # recalc distances
            for idx in range(len(routes)):
                dists[idx] = compute_route_distance(routes[idx])
            total_dist = sum(dists)
            max_dist = max(dists)
            # Update best if improved
            if max_dist < best_max_dist - 1e-12 or (abs(max_dist - best_max_dist) < 1e-12 and total_dist < best_total_dist - 1e-12):
                best_max_dist = max_dist
                best_total_dist = total_dist
                best_routes = [route[:] for route in routes]
            report_best_vrp(routes)
        else:
            # last restart, keep result
            if best_routes is None or max_dist < best_max_dist - 1e-12 or (abs(max_dist - best_max_dist) < 1e-12 and total_dist < best_total_dist - 1e-12):
                best_max_dist = max_dist
                best_total_dist = total_dist
                best_routes = [route[:] for route in routes]

    return best_routes if best_routes is not None else routes