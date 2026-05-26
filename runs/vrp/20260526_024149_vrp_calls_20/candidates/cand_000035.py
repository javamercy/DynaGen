import numpy as np
import random

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def two_opt(route, dm):
    improved = True
    best_route = route[:]
    best_dist = route_distance(best_route, dm)
    while improved:
        improved = False
        for i in range(1, len(best_route)-2):
            for j in range(i+1, len(best_route)-1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_dist = route_distance(new_route, dm)
                if new_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    return best_route

def adapt_perturb(routes, truck_count, n, perturbation_size):
    num_cust = min(perturbation_size, n-1)
    customers = list(range(1, n))
    random.shuffle(customers)
    selected = customers[:num_cust]
    for cust in selected:
        for idx, route in enumerate(routes):
            if cust in route:
                pos = route.index(cust)
                routes[idx] = route[:pos] + route[pos+1:]
                break
    for cust in selected:
        r_idx = random.randint(0, truck_count-1)
        route = routes[r_idx]
        insert_pos = random.randint(1, len(route)-1)
        routes[r_idx] = route[:insert_pos] + [cust] + route[insert_pos:]
    return routes

def shake(routes, truck_count, n, dm):
    dists = [route_distance(r, dm) for r in routes]
    max_idx = dists.index(max(dists))
    if len(routes[max_idx]) <= 3:
        return routes
    max_len = len(routes[max_idx])
    block_size = min(3, max_len - 2)
    start = random.randint(1, max_len - block_size - 1)
    block = routes[max_idx][start:start+block_size]
    new_longest = routes[max_idx][:start] + routes[max_idx][start+block_size:]
    routes[max_idx] = new_longest
    for cust in block:
        r_idx = random.randint(0, truck_count-1)
        route = routes[r_idx]
        insert_pos = random.randint(1, len(route)-1)
        routes[r_idx] = route[:insert_pos] + [cust] + route[insert_pos:]
    return routes

def or_opt(routes, truck_count, dm):
    # Try moving a block of up to 3 consecutive customers from longest route to others
    dists = [route_distance(r, dm) for r in routes]
    max_idx = dists.index(max(dists))
    if len(routes[max_idx]) <= 3:
        return False, routes
    best_routes = None
    best_new_max = float('inf')
    improved = False
    max_len = len(routes[max_idx])
    max_block = min(3, max_len - 2)
    for block_size in range(1, max_block+1):
        for start in range(1, max_len - block_size):
            block = routes[max_idx][start:start+block_size]
            new_max_route = routes[max_idx][:start] + routes[max_idx][start+block_size:]
            new_max_dist = route_distance(new_max_route, dm)
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for insert_pos in range(1, len(other_route)):
                    new_other_route = other_route[:insert_pos] + block + other_route[insert_pos:]
                    new_other_dist = route_distance(new_other_route, dm)
                    new_dists = dists.copy()
                    new_dists[max_idx] = new_max_dist
                    new_dists[other_idx] = new_other_dist
                    new_max = max(new_dists)
                    if new_max < best_new_max - 1e-9:
                        best_new_max = new_max
                        best_routes = routes.copy()
                        best_routes[max_idx] = new_max_route
                        best_routes[other_idx] = new_other_route
                        improved = True
    if improved and best_new_max < max(dists) - 1e-9:
        for i in range(truck_count):
            routes[i] = best_routes[i]
        return True, routes
    else:
        return False, routes

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= len(customers):
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # Initialization: each customer as its own route
    routes = [[0, c, 0] for c in customers]

    # Clarke-Wright savings merging
    while len(routes) > truck_count:
        best_saving = -1e9
        best_pair = None
        best_order = 0
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                ri = routes[i]
                rj = routes[j]
                if len(ri) <= 2 or len(rj) <= 2:
                    continue
                last_i = ri[-2]
                first_i = ri[1]
                last_j = rj[-2]
                first_j = rj[1]
                s1 = distance_matrix[0][last_i] + distance_matrix[0][first_j] - distance_matrix[last_i][first_j]
                s2 = distance_matrix[0][last_j] + distance_matrix[0][first_i] - distance_matrix[last_j][first_i]
                if s1 > best_saving:
                    best_saving = s1
                    best_pair = (i, j)
                    best_order = 0
                if s2 > best_saving:
                    best_saving = s2
                    best_pair = (i, j)
                    best_order = 1
        if best_pair is None:
            break
        i, j = best_pair
        if best_order == 0:
            new_route = routes[i][:-1] + routes[j][1:]
        else:
            new_route = routes[j][:-1] + routes[i][1:]
        if i < j:
            del routes[j]
            del routes[i]
        else:
            del routes[i]
            del routes[j]
        routes.append(new_route)

    dists = [route_distance(r, distance_matrix) for r in routes]
    best_routes = [list(r) for r in routes]
    best_max = max(dists)
    report_best_vrp(best_routes)

    random.seed(0)
    max_restarts = min(5, n * truck_count // 5 + 1)
    current_routes = [list(r) for r in routes]

    stagnation_counter = 0
    max_stagnation = 3
    base_perturb_size = 3
    max_perturb_size = 10
    perturb_size = base_perturb_size

    for restart in range(max_restarts):
        if restart > 0:
            current_routes = adapt_perturb(current_routes, truck_count, n, perturb_size)
        # Shake occasionally
        if restart % 2 == 1 and restart > 0:
            current_routes = shake(current_routes, truck_count, n, distance_matrix)

        max_iter = n * truck_count
        for iteration in range(max_iter):
            dists = [route_distance(r, distance_matrix) for r in current_routes]
            max_dist = max(dists)
            if max_dist < best_max - 1e-9:
                best_max = max_dist
                best_routes = [list(r) for r in current_routes]
                report_best_vrp(best_routes)
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Adaptive perturbation size
            if stagnation_counter >= max_stagnation:
                perturb_size = min(perturb_size + 1, max_perturb_size)
                stagnation_counter = 0
            else:
                perturb_size = base_perturb_size

            max_idx = dists.index(max_dist)
            improved = False

            # Relocate moves from longest route
            if len(current_routes[max_idx]) > 2:
                for pos in range(1, len(current_routes[max_idx])-1):
                    cust = current_routes[max_idx][pos]
                    new_max_route = current_routes[max_idx][:pos] + current_routes[max_idx][pos+1:]
                    new_max_dist = route_distance(new_max_route, distance_matrix)
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = current_routes[other_idx]
                        for insert_pos in range(1, len(other_route)):
                            new_other_route = other_route[:insert_pos] + [cust] + other_route[insert_pos:]
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            if new_max < max_dist - 1e-9:
                                current_routes[max_idx] = new_max_route
                                current_routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

            # If no relocate improvement, try swap moves
            if not improved and len(current_routes[max_idx]) > 2:
                for other_idx in range(truck_count):
                    if other_idx == max_idx or len(current_routes[other_idx]) <= 2:
                        continue
                    for pos_max in range(1, len(current_routes[max_idx])-1):
                        cust_a = current_routes[max_idx][pos_max]
                        for pos_other in range(1, len(current_routes[other_idx])-1):
                            cust_b = current_routes[other_idx][pos_other]
                            new_max_route = current_routes[max_idx].copy()
                            new_max_route[pos_max] = cust_b
                            new_max_dist = route_distance(new_max_route, distance_matrix)
                            new_other_route = current_routes[other_idx].copy()
                            new_other_route[pos_other] = cust_a
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            if new_max < max_dist - 1e-9:
                                current_routes[max_idx] = new_max_route
                                current_routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

            # Try 2-opt on longest route if no improvement
            if not improved and len(current_routes[max_idx]) > 2:
                new_route = two_opt(current_routes[max_idx], distance_matrix)
                new_dist = route_distance(new_route, distance_matrix)
                if new_dist < dists[max_idx] - 1e-9:
                    current_routes[max_idx] = new_route
                    improved = True

            # Try Or-opt cross-route exchange
            if not improved:
                improved_or, current_routes = or_opt(current_routes, truck_count, distance_matrix)
                if improved_or:
                    improved = True

            # Epsilon acceptance: accept if new max <= old max + small threshold
            if not improved:
                if len(current_routes[max_idx]) > 2:
                    pos = random.randint(1, len(current_routes[max_idx])-2)
                    cust = current_routes[max_idx][pos]
                    new_max_route = current_routes[max_idx][:pos] + current_routes[max_idx][pos+1:]
                    new_max_dist = route_distance(new_max_route, distance_matrix)
                    other_idx = random.choice([i for i in range(truck_count) if i != max_idx])
                    other_route = current_routes[other_idx]
                    insert_pos = random.randint(1, len(other_route)-1)
                    new_other_route = other_route[:insert_pos] + [cust] + other_route[insert_pos:]
                    new_other_dist = route_distance(new_other_route, distance_matrix)
                    new_dists = dists.copy()
                    new_dists[max_idx] = new_max_dist
                    new_dists[other_idx] = new_other_dist
                    new_max = max(new_dists)
                    epsilon = 0.02 * max_dist
                    if new_max <= max_dist + epsilon:
                        current_routes[max_idx] = new_max_route
                        current_routes[other_idx] = new_other_route
                        improved = True

            if not improved:
                break

        final_dists = [route_distance(r, distance_matrix) for r in current_routes]
        current_max = max(final_dists)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)

    report_best_vrp(best_routes)
    return best_routes