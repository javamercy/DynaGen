import math
import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0, 0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    def compute_route_distance(route):
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i], route[i+1]]
        return dist

    def compute_total_and_max(routes):
        distances = [compute_route_distance(r) for r in routes]
        return sum(distances), max(distances)

    def report_if_better(routes, best_routes, best_max, best_total):
        total_dist, max_dist = compute_total_and_max(routes)
        if max_dist < best_max[0] or (abs(max_dist - best_max[0]) < 1e-12 and total_dist < best_total[0]):
            best_routes[:] = [r[:] for r in routes]
            best_max[0] = max_dist
            best_total[0] = total_dist
            report_best_vrp(routes)

    def local_search(routes):
        n_routes = len(routes)
        max_iter = n * truck_count
        for _ in range(max_iter):
            improved = False
            # intra-route 2-opt
            for i in range(n_routes):
                route = routes[i]
                if len(route) <= 3:
                    continue
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        if b - a == 1:
                            continue
                        old_cost = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                        new_cost = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                        if new_cost < old_cost - 1e-12:
                            route[a:b+1] = route[a:b+1][::-1]
                            improved = True
            # inter-route relocate and swap
            for i in range(n_routes):
                for j in range(n_routes):
                    if i == j:
                        continue
                    route_i = routes[i]
                    route_j = routes[j]
                    if len(route_i) <= 2:
                        continue
                    # relocate: move customer from route_i to route_j
                    for pos_i in range(1, len(route_i)-1):
                        customer = route_i[pos_i]
                        # evaluate removal
                        prev_i = route_i[pos_i-1]
                        next_i = route_i[pos_i+1]
                        removal_saving = distance_matrix[prev_i, customer] + distance_matrix[customer, next_i] - distance_matrix[prev_i, next_i]
                        new_dist_i = compute_route_distance(route_i) - removal_saving
                        # evaluate insertion in route_j
                        best_insert_cost = math.inf
                        best_pos = None
                        for k in range(1, len(route_j)):
                            pred = route_j[k-1]
                            succ = route_j[k]
                            insert_cost = distance_matrix[pred, customer] + distance_matrix[customer, succ] - distance_matrix[pred, succ]
                            if insert_cost < best_insert_cost - 1e-12:
                                best_insert_cost = insert_cost
                                best_pos = k
                        if best_pos is None:
                            continue
                        new_dist_j = compute_route_distance(route_j) + best_insert_cost
                        # check improvement in max
                        old_max = max(compute_route_distance(r) for r in routes)
                        new_max = max(new_dist_i, new_dist_j, *[compute_route_distance(routes[k]) for k in range(n_routes) if k not in (i,j)])
                        if new_max < old_max - 1e-12 or (abs(new_max - old_max) < 1e-12 and (new_dist_i + new_dist_j) < (compute_route_distance(route_i) + compute_route_distance(route_j)) - 1e-12):
                            # perform relocate
                            del route_i[pos_i]
                            route_j.insert(best_pos, customer)
                            improved = True
                            break
                    if improved:
                        break
                    # swap customers between routes
                    for pos_i in range(1, len(route_i)-1):
                        for pos_j in range(1, len(route_j)-1):
                            cust_i = route_i[pos_i]
                            cust_j = route_j[pos_j]
                            # compute new distances after swap
                            # route_i
                            prev_i = route_i[pos_i-1]
                            next_i = route_i[pos_i+1]
                            old_i = distance_matrix[prev_i, cust_i] + distance_matrix[cust_i, next_i]
                            new_i = distance_matrix[prev_i, cust_j] + distance_matrix[cust_j, next_i]
                            new_dist_i = compute_route_distance(route_i) - old_i + new_i
                            # route_j
                            prev_j = route_j[pos_j-1]
                            next_j = route_j[pos_j+1]
                            old_j = distance_matrix[prev_j, cust_j] + distance_matrix[cust_j, next_j]
                            new_j = distance_matrix[prev_j, cust_i] + distance_matrix[cust_i, next_j]
                            new_dist_j = compute_route_distance(route_j) - old_j + new_j
                            old_max = max(compute_route_distance(r) for r in routes)
                            new_max = max(new_dist_i, new_dist_j, *[compute_route_distance(routes[k]) for k in range(n_routes) if k not in (i,j)])
                            if new_max < old_max - 1e-12 or (abs(new_max - old_max) < 1e-12 and (new_dist_i + new_dist_j) < (compute_route_distance(route_i) + compute_route_distance(route_j)) - 1e-12):
                                route_i[pos_i] = cust_j
                                route_j[pos_j] = cust_i
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    def perturb(routes, temperature):
        n_routes = len(routes)
        # choose two different non-empty routes
        nonempty = [i for i, r in enumerate(routes) if len(r) > 2]
        if len(nonempty) < 2:
            return
        i = random.choice(nonempty)
        j = random.choice([x for x in nonempty if x != i])
        if random.random() < 0.5:
            # swap a customer between i and j
            pos_i = random.randint(1, len(routes[i])-2)
            pos_j = random.randint(1, len(routes[j])-2)
            cust_i = routes[i][pos_i]
            cust_j = routes[j][pos_j]
            # compute new distances for acceptance check (but we do not check here)
            routes[i][pos_i] = cust_j
            routes[j][pos_j] = cust_i
        else:
            # relocate a customer from i to j
            pos_i = random.randint(1, len(routes[i])-2)
            customer = routes[i][pos_i]
            del routes[i][pos_i]
            pos_j = random.randint(1, len(routes[j])-1)
            routes[j].insert(pos_j, customer)

    def greedy_initial():
        # start with each customer as a separate route, then merge
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) > truck_count:
            best_score = math.inf
            best_pair = None
            best_merged = None
            best_new_dist = None
            for i in range(len(routes)):
                for j in range(len(routes)):
                    if i == j:
                        continue
                    r_i = routes[i]
                    r_j = routes[j]
                    # try merging: r_i + r_j (excluding depot) or r_j + r_i
                    # consider both orders
                    # order1: r_i[:-1] + r_j[1:]
                    last_i = r_i[-2]
                    first_j = r_j[1]
                    dist1 = compute_route_distance(r_i) + compute_route_distance(r_j) - distance_matrix[last_i, 0] - distance_matrix[0, first_j] + distance_matrix[last_i, first_j]
                    merged1 = r_i[:-1] + r_j[1:]
                    # order2: r_j[:-1] + r_i[1:]
                    last_j = r_j[-2]
                    first_i = r_i[1]
                    dist2 = compute_route_distance(r_i) + compute_route_distance(r_j) - distance_matrix[last_j, 0] - distance_matrix[0, first_i] + distance_matrix[last_j, first_i]
                    merged2 = r_j[:-1] + r_i[1:]
                    # choose order with smaller distance
                    if dist1 <= dist2:
                        merged = merged1
                        new_dist = dist1
                    else:
                        merged = merged2
                        new_dist = dist2
                    # score: weighted sum of max distance and total distance (alpha from list, but here we use simple: prefer smaller max)
                    # we'll use alpha=0.9 focusing on max
                    alpha = 0.9
                    new_max = max(new_dist, max(compute_route_distance(routes[k]) for k in range(len(routes)) if k not in (i,j)))
                    score = alpha * new_max + (1-alpha) * new_dist
                    if score < best_score - 1e-12:
                        best_score = score
                        best_pair = (i, j)
                        best_merged = merged
                        best_new_dist = new_dist
            if best_pair is None:
                break
            i, j = best_pair
            routes[i] = best_merged
            del routes[j]
        # fill empty routes if needed
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    best_routes = None
    best_max = [math.inf]
    best_total = [math.inf]
    # multiple starts
    for restart in range(5):
        routes = greedy_initial()
        # initial local search
        routes = local_search(routes)
        current_max, current_total = compute_total_and_max(routes)
        report_if_better(routes, best_routes, best_max, best_total)
        # ILS with simulated annealing
        T0 = 10.0
        T_min = 1e-3
        cooling_rate = 0.95
        T = T0
        ils_iter = n * truck_count
        for it in range(ils_iter):
            old_routes = [r[:] for r in routes]
            old_total, old_max = current_total, current_max
            perturb(routes, T)
            routes = local_search(routes)
            new_total, new_max = compute_total_and_max(routes)
            # simulated annealing acceptance
            delta = new_max - old_max
            if delta < 0 or random.random() < math.exp(-delta / T):
                # accept
                current_max, current_total = new_max, new_total
            else:
                # revert
                routes = old_routes
                current_max, current_total = old_max, old_total
            # update best
            report_if_better(routes, best_routes, best_max, best_total)
            T = max(T * cooling_rate, T_min)
    # ensure best_routes is assigned
    if best_routes is None:
        best_routes = routes
    return best_routes