def solve_vrp(distance_matrix, truck_count):
    import math
    import random
    import numpy as np
    random.seed(0)
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0,0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    #--- helper functions ---
    def compute_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    #--- construction: greedy merge minimizing max distance ---
    routes = [[0, i, 0] for i in range(1, n)]
    dists = [2 * distance_matrix[0, i] for i in range(1, n)]
    current_max = max(dists) if dists else 0.0
    total_dist = sum(dists)

    while len(routes) > truck_count:
        best_new_max = math.inf
        best_new_total = math.inf
        best_pair = None
        best_merged_route = None
        best_merged_dist = None
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                r_i = routes[i]
                r_j = routes[j]
                # i then j
                last_i = r_i[-2]
                first_j = r_j[1]
                dist_ij = dists[i] + dists[j] - distance_matrix[last_i, 0] - distance_matrix[0, first_j] + distance_matrix[last_i, first_j]
                # j then i
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
                if (new_max < best_new_max) or (new_max == best_new_max and new_dist < best_new_total):
                    best_new_max = new_max
                    best_new_total = new_dist
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

    while len(routes) < truck_count:
        routes.append([0,0])
        dists.append(0.0)

    # initial best
    best_routes = [list(r) for r in routes]
    best_dists = list(dists)
    best_max = max(dists)
    best_total = sum(dists)
    report_best_vrp(routes)

    #--- improvement functions (2-opt and relocate/swap) ---
    def improve(routes, dists, max_dist, total_dist):
        # intra-route 2-opt
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
                            max_dist = max(dists)
                            total_dist = sum(dists)
                            yield (routes, dists, max_dist, total_dist)
                            break
                    if improved:
                        break
        # inter-route relocate and swap (best improvement)
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
                # relocate
                for pos in range(1, len(route_i)-1):
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
                # swap
                for j_route in range(len(routes)):
                    if j_route == i_route:
                        continue
                    route_j = routes[j_route]
                    for pos_i in range(1, len(route_i)-1):
                        cust_i = route_i[pos_i]
                        for pos_j in range(1, len(route_j)-1):
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

    # initial improvement
    for _ in improve(routes, dists, max_dist, total_dist):
        max_dist = max(dists)
        total_dist = sum(dists)
        report_best_vrp(routes)
        if max(dists) < best_max - 1e-12 or (abs(max(dists)-best_max)<1e-12 and sum(dists) < best_total - 1e-12):
            best_max = max(dists)
            best_total = sum(dists)
            best_routes = [list(r) for r in routes]
            best_dists = list(dists)

    #--- multi-perturbation with SA acceptance ---
    max_restarts = 10
    temperature = 0.1 * best_max  # initial temperature
    cooling = temperature / max_restarts
    for restart in range(max_restarts):
        # start from best solution
        routes = [list(r) for r in best_routes]
        dists = list(best_dists)
        max_dist = best_max
        total_dist = best_total

        # ruin-and-recreate on multiple routes
        # select up to 3 routes (or less) that have customers
        candidate_routes = [i for i, r in enumerate(routes) if len(r) > 2]
        if len(candidate_routes) < 2:
            # if not enough, just perturb one route
            selected = [candidate_routes[0]] if candidate_routes else []
        else:
            num_perturb = random.randint(1, min(3, len(candidate_routes)))
            selected = random.sample(candidate_routes, num_perturb)
        removed = []
        for idx in selected:
            route = routes[idx]
            if len(route) <= 3:
                continue
            num_remove = random.randint(1, min(len(route)-2, 2))  # remove small number
            positions = random.sample(range(1, len(route)-1), num_remove)
            for pos in sorted(positions, reverse=True):
                removed.append((idx, route.pop(pos)))
            dists[idx] = compute_dist(route)
        # reinsert all removed customers greedily
        for route_idx, cust in removed:
            best_route_idx = None
            best_insert_pos = None
            best_new_max = math.inf
            best_new_total = math.inf
            for i_route in range(len(routes)):
                route_j = routes[i_route]
                for k in range(1, len(route_j)):
                    pred = route_j[k-1]
                    succ = route_j[k]
                    insert_cost = distance_matrix[pred, cust] + distance_matrix[cust, succ] - distance_matrix[pred, succ]
                    new_dist_j = dists[i_route] + insert_cost
                    new_max = max([dists[idx] if idx != i_route else new_dist_j for idx in range(len(routes))])
                    new_total = total_dist + insert_cost
                    if (new_max < best_new_max - 1e-12) or (abs(new_max - best_new_max) < 1e-12 and new_total < best_new_total - 1e-12):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_route_idx = i_route
                        best_insert_pos = k
                        best_new_dist = new_dist_j
            if best_route_idx is not None:
                routes[best_route_idx].insert(best_insert_pos, cust)
                dists[best_route_idx] = best_new_dist
                total_dist = best_new_total
                max_dist = best_new_max
            else:
                # fallback: create new route? but unlikely, reinsert anyway
                # create a new route with a new truck? But we have fixed truck count, so skip
                pass
        # improvement after perturbation
        for _ in improve(routes, dists, max_dist, total_dist):
            max_dist = max(dists)
            total_dist = sum(dists)
            report_best_vrp(routes)
        # update best or accept with SA criterion
        candidate_max = max(dists)
        candidate_total = sum(dists)
        if candidate_max < best_max - 1e-12 or (abs(candidate_max - best_max) < 1e-12 and candidate_total < best_total - 1e-12):
            best_max = candidate_max
            best_total = candidate_total
            best_routes = [list(r) for r in routes]
            best_dists = list(dists)
        else:
            delta = candidate_max - best_max
            if delta > 0 and random.random() < math.exp(-delta / temperature):
                # accept worse solution for next restart
                best_max = candidate_max
                best_total = candidate_total
                best_routes = [list(r) for r in routes]
                best_dists = list(dists)
        temperature -= cooling
        if temperature < 0:
            temperature = 0.0

    report_best_vrp(best_routes)
    return best_routes