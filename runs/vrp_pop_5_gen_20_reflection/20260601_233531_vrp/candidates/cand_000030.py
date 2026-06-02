def solve_vrp(distance_matrix, truck_count):
    import math
    import random
    import numpy as np
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0, 0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    customers = list(range(1, n))
    
    def initial_routes():
        routes = [[0, c, 0] for c in customers]
        dists = [2 * distance_matrix[0, c] for c in customers]
        current_max = max(dists)
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
            if best_pair is None or best_merged_route is None:
                break
            i, j = best_pair
            routes[i] = best_merged_route
            dists[i] = best_merged_dist
            current_max = best_new_max
            del routes[j]
            del dists[j]
        while len(routes) < truck_count:
            routes.append([0, 0])
            dists.append(0.0)
        return routes, dists

    def evaluate(routes):
        dists = []
        for route in routes:
            d = 0.0
            for k in range(len(route)-1):
                d += distance_matrix[route[k], route[k+1]]
            dists.append(d)
        return dists, max(dists), sum(dists)

    def local_search(routes, dists, max_dist, total_dist):
        improved = True
        max_iter = n * truck_count
        for iteration in range(max_iter):
            max_route_idx = None
            max_dist_val = -1.0
            for idx, d in enumerate(dists):
                if d > max_dist_val + 1e-12:
                    max_dist_val = d
                    max_route_idx = idx
            if max_route_idx is None or max_dist_val <= 0:
                break

            best_improvement = None
            best_new_max = max_dist_val
            best_new_total = total_dist

            route_long = routes[max_route_idx]
            for pos in range(1, len(route_long) - 1):
                customer = route_long[pos]
                prev = route_long[pos-1]
                nxt = route_long[pos+1]
                removal_saving = distance_matrix[prev, customer] + distance_matrix[customer, nxt] - distance_matrix[prev, nxt]
                new_long_dist = max_dist_val - removal_saving
                for target_idx in range(len(routes)):
                    if target_idx == max_route_idx:
                        continue
                    target_route = routes[target_idx]
                    best_insert_cost = math.inf
                    best_insert_pos = None
                    for k in range(1, len(target_route)):
                        pred = target_route[k-1]
                        succ = target_route[k] if k < len(target_route) else 0
                        insert_cost = distance_matrix[pred, customer] + distance_matrix[customer, succ] - distance_matrix[pred, succ]
                        if insert_cost < best_insert_cost:
                            best_insert_cost = insert_cost
                            best_insert_pos = k
                    new_target_dist = dists[target_idx] + best_insert_cost
                    other_dists = [dists[idx] for idx in range(len(dists)) if idx not in (max_route_idx, target_idx)]
                    combined = other_dists + [new_long_dist, new_target_dist]
                    candidate_max = max(combined)
                    candidate_total = total_dist - removal_saving + best_insert_cost
                    if (candidate_max < best_new_max - 1e-12) or (abs(candidate_max - best_new_max) < 1e-12 and candidate_total < best_new_total):
                        best_new_max = candidate_max
                        best_new_total = candidate_total
                        best_improvement = ('relocate', max_route_idx, pos, target_idx, best_insert_pos, new_long_dist, new_target_dist, removal_saving, best_insert_cost)

            for target_idx in range(len(routes)):
                if target_idx == max_route_idx:
                    continue
                route_target = routes[target_idx]
                for pos_long in range(1, len(route_long) - 1):
                    cust_long = route_long[pos_long]
                    for pos_target in range(1, len(route_target) - 1):
                        cust_target = route_target[pos_target]
                        prev_long = route_long[pos_long-1]
                        next_long = route_long[pos_long+1]
                        saving_long = distance_matrix[prev_long, cust_long] + distance_matrix[cust_long, next_long] - distance_matrix[prev_long, next_long]
                        prev_target = route_target[pos_target-1]
                        next_target = route_target[pos_target+1]
                        saving_target = distance_matrix[prev_target, cust_target] + distance_matrix[cust_target, next_target] - distance_matrix[prev_target, next_target]
                        add_long = distance_matrix[prev_long, cust_target] + distance_matrix[cust_target, next_long] - distance_matrix[prev_long, next_long]
                        new_long_dist = max_dist_val - saving_long + add_long
                        add_target = distance_matrix[prev_target, cust_long] + distance_matrix[cust_long, next_target] - distance_matrix[prev_target, next_target]
                        new_target_dist = dists[target_idx] - saving_target + add_target
                        other_dists = [dists[idx] for idx in range(len(dists)) if idx not in (max_route_idx, target_idx)]
                        combined = other_dists + [new_long_dist, new_target_dist]
                        candidate_max = max(combined)
                        candidate_total = total_dist - saving_long + add_long - saving_target + add_target
                        if (candidate_max < best_new_max - 1e-12) or (abs(candidate_max - best_new_max) < 1e-12 and candidate_total < best_new_total):
                            best_new_max = candidate_max
                            best_new_total = candidate_total
                            best_improvement = ('swap', max_route_idx, pos_long, target_idx, pos_target, new_long_dist, new_target_dist, saving_long, add_long, saving_target, add_target)

            if best_improvement is None:
                break
            if best_improvement[0] == 'relocate':
                _, i_route, pos, j_route, insert_pos, new_dist_i, new_dist_j, saving, add = best_improvement
                customer = routes[i_route].pop(pos)
                dists[i_route] = new_dist_i
                routes[j_route].insert(insert_pos, customer)
                dists[j_route] = new_dist_j
            else:
                _, i_route, pos_i, j_route, pos_j, new_dist_i, new_dist_j, sav_i, add_i, sav_j, add_j = best_improvement
                cust_i = routes[i_route][pos_i]
                cust_j = routes[j_route][pos_j]
                routes[i_route][pos_i] = cust_j
                routes[j_route][pos_j] = cust_i
                dists[i_route] = new_dist_i
                dists[j_route] = new_dist_j
            total_dist = best_new_total
            max_dist = best_new_max
            report_best_vrp(routes)
        return routes, dists, max_dist, total_dist

    # Initial solution
    routes, dists = initial_routes()
    dists, max_dist, total_dist = evaluate(routes)
    report_best_vrp(routes)
    routes, dists, max_dist, total_dist = local_search(routes, dists, max_dist, total_dist)
    best_routes = [r[:] for r in routes]
    best_max = max_dist

    # Perturbation and restart loop (max 3 rounds)
    for restart in range(3):
        # Perturb: randomly relocate 1 customer from the longest route to a random other route
        max_idx = max(range(len(dists)), key=lambda i: dists[i])
        if len(routes[max_idx]) <= 2:
            break
        other_indices = [i for i in range(len(routes)) if i != max_idx]
        if not other_indices:
            break
        # Randomly choose a customer from longest route
        pos = random.randint(1, len(routes[max_idx])-2)
        customer = routes[max_idx].pop(pos)
        # Remove from dists placeholder
        target_idx = random.choice(other_indices)
        target_route = routes[target_idx]
        insert_pos = random.randint(1, len(target_route)-1)
        target_route.insert(insert_pos, customer)
        # Not recalculating dists exactly, but local_search will fix
        dists, max_dist, total_dist = evaluate(routes)
        report_best_vrp(routes)
        routes, dists, max_dist, total_dist = local_search(routes, dists, max_dist, total_dist)
        if max_dist < best_max - 1e-12:
            best_max = max_dist
            best_routes = [r[:] for r in routes]

    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0,0])
    report_best_vrp(best_routes)
    return best_routes