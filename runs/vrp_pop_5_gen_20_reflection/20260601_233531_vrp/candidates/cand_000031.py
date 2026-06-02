def solve_vrp(distance_matrix, truck_count):
    import math
    import random
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

    # Construction: regret-2 insertion to minimize max route distance
    unvisited = set(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    dists = [0.0] * truck_count
    # Initialize each route with one customer to avoid empty routes
    customers = list(unvisited)
    random.shuffle(customers)
    for i in range(truck_count):
        if not customers:
            break
        cust = customers.pop()
        unvisited.remove(cust)
        routes[i] = [0, cust, 0]
        dists[i] = compute_dist(routes[i])
    # Insert remaining customers
    while unvisited:
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_max = math.inf
        best_delta = 0
        for cust in list(unvisited):
            # Compute best insertion cost and route for this customer
            best_cost = math.inf
            best_route = None
            best_pos_local = None
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    pred = route[pos-1]
                    succ = route[pos]
                    new_dist = dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, cust] + distance_matrix[cust, succ]
                    if new_dist < best_cost:
                        best_cost = new_dist
                        best_route = r_idx
                        best_pos_local = pos
            # Regret-2: difference between best and second best insertion cost
            if best_route is None:
                continue
            second_best_cost = math.inf
            for r_idx, route in enumerate(routes):
                if r_idx == best_route:
                    continue
                for pos in range(1, len(route)):
                    pred = route[pos-1]
                    succ = route[pos]
                    new_dist = dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, cust] + distance_matrix[cust, succ]
                    if new_dist < second_best_cost:
                        second_best_cost = new_dist
            if second_best_cost == math.inf:
                second_best_cost = best_cost
            regret = second_best_cost - best_cost
            # Compute resulting max distance if inserted in best route
            new_dists = [dists[i] if i != best_route else best_cost for i in range(truck_count)]
            new_max = max(new_dists)
            # Prefer larger regret, then smaller new_max, then smaller best_cost
            if (regret > best_delta) or (regret == best_delta and new_max < best_max) or (regret == best_delta and new_max == best_max and best_cost < best_delta):
                best_delta = regret
                best_cust = cust
                best_route_idx = best_route
                best_pos = best_pos_local
                best_max = new_max
        if best_cust is None:
            break
        # Insert best customer
        route = routes[best_route_idx]
        route.insert(best_pos, best_cust)
        dists[best_route_idx] += distance_matrix[route[best_pos-1], best_cust] + distance_matrix[best_cust, route[best_pos+1]] - distance_matrix[route[best_pos-1], route[best_pos+1]]
        unvisited.remove(best_cust)

    # If any route empty (shouldn't happen), fill with [0,0]
    for i in range(truck_count):
        if len(routes[i]) == 2:
            routes[i] = [0, 0]
            dists[i] = 0.0

    total_dist = sum(dists)
    max_dist = max(dists)
    best_routes = [list(r) for r in routes]
    best_dists = list(dists)
    best_max = max_dist
    best_total = total_dist

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
                # Relocate moves
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
                # Swap moves
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
        max_dist = max(dists)
        total_dist = sum(dists)
        report_best_vrp(routes)
        if max_dist < best_max - 1e-12 or (abs(max_dist - best_max) < 1e-12 and total_dist < best_total - 1e-12):
            best_max = max_dist
            best_total = total_dist
            best_routes = [list(r) for r in routes]
            best_dists = list(dists)

    for _ in improve(routes, dists, max_dist, total_dist):
        report_best_vrp(routes)
        if max(dists) < best_max - 1e-12 or (abs(max(dists)-best_max) < 1e-12 and sum(dists) < best_total - 1e-12):
            best_max = max(dists)
            best_total = sum(dists)
            best_routes = [list(r) for r in routes]
            best_dists = list(dists)

    # Perturbation and restart loop
    max_restarts = 10
    for restart in range(max_restarts):
        routes = [list(r) for r in best_routes]
        dists = list(best_dists)
        max_dist = best_max
        total_dist = best_total

        # Perturb: random swap of two customers from different routes
        routes_with_cust = [i for i, r in enumerate(routes) if len(r) > 2]
        if len(routes_with_cust) >= 2:
            i_route, j_route = random.sample(routes_with_cust, 2)
            pos_i = random.randint(1, len(routes[i_route])-2)
            pos_j = random.randint(1, len(routes[j_route])-2)
            cust_i = routes[i_route][pos_i]
            cust_j = routes[j_route][pos_j]
            routes[i_route][pos_i], routes[j_route][pos_j] = cust_j, cust_i
            dists[i_route] = compute_dist(routes[i_route])
            dists[j_route] = compute_dist(routes[j_route])
            max_dist = max(dists)
            total_dist = sum(dists)
            report_best_vrp(routes)
        else:
            pass

        # Re-optimize after perturbation
        for _ in apply_2opt(routes, dists):
            max_dist = max(dists)
            total_dist = sum(dists)
            report_best_vrp(routes)
            if max_dist < best_max - 1e-12 or (abs(max_dist - best_max) < 1e-12 and total_dist < best_total - 1e-12):
                best_max = max_dist
                best_total = total_dist
                best_routes = [list(r) for r in routes]
                best_dists = list(dists)

        for _ in improve(routes, dists, max_dist, total_dist):
            report_best_vrp(routes)
            if max(dists) < best_max - 1e-12 or (abs(max(dists)-best_max) < 1e-12 and sum(dists) < best_total - 1e-12):
                best_max = max(dists)
                best_total = sum(dists)
                best_routes = [list(r) for r in routes]
                best_dists = list(dists)

    report_best_vrp(best_routes)
    return best_routes