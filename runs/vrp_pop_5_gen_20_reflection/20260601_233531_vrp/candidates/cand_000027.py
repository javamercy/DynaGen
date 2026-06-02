import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0, 0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    customers = list(range(1, n))
    routes = [[0, c, 0] for c in customers]
    dists = [2 * distance_matrix[0, c] for c in customers]

    # Construction: merge routes to reduce count to truck_count
    while len(routes) > truck_count:
        best_new_max = math.inf
        best_new_dist = math.inf
        best_pair = None
        best_orientation = None
        current_max = max(dists)
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue
                # orientation: i then j
                r1 = routes[i]
                r2 = routes[j]
                dist_ij = dists[i] + dists[j] - distance_matrix[r1[-2], 0] - distance_matrix[0, r2[1]] + distance_matrix[r1[-2], r2[1]]
                new_max_ij = max(current_max, dist_ij)
                # orientation: j then i
                dist_ji = dists[i] + dists[j] - distance_matrix[r2[-2], 0] - distance_matrix[0, r1[1]] + distance_matrix[r2[-2], r1[1]]
                new_max_ji = max(current_max, dist_ji)
                if new_max_ij < best_new_max or (new_max_ij == best_new_max and dist_ij < best_new_dist):
                    best_new_max = new_max_ij
                    best_new_dist = dist_ij
                    best_pair = (i, j)
                    best_orientation = 'ij'
                if new_max_ji < best_new_max or (new_max_ji == best_new_max and dist_ji < best_new_dist):
                    best_new_max = new_max_ji
                    best_new_dist = dist_ji
                    best_pair = (i, j)
                    best_orientation = 'ji'
        if best_pair is None:
            break
        i, j = best_pair
        if best_orientation == 'ij':
            merged = routes[i][:-1] + routes[j][1:]
        else:
            merged = routes[j][:-1] + routes[i][1:]
        routes[i] = merged
        dists[i] = best_new_dist
        del routes[j]
        del dists[j]

    # Add empty routes if under truck_count
    while len(routes) < truck_count:
        routes.append([0, 0])
        dists.append(0.0)

    total_dist = sum(dists)
    report_best_vrp(routes)
    max_dist = max(dists)

    # Local search
    def local_search():
        nonlocal routes, dists, total_dist, max_dist
        max_iter = n * truck_count
        for _ in range(max_iter):
            # Find longest route
            max_route_idx = None
            max_dist_val = -1
            for idx, d in enumerate(dists):
                if d > max_dist_val:
                    max_dist_val = d
                    max_route_idx = idx
            if max_route_idx is None or max_dist_val == 0:
                break
            route = routes[max_route_idx]
            best_improvement = None
            best_new_max = max_dist_val
            best_new_total = total_dist
            for pos in range(1, len(route) - 1):
                customer = route[pos]
                prev = route[pos - 1]
                next = route[pos + 1]
                removal_saving = distance_matrix[prev, customer] + distance_matrix[customer, next] - distance_matrix[prev, next]
                new_dist_removed = dists[max_route_idx] - removal_saving
                for target_idx in range(len(routes)):
                    if target_idx == max_route_idx:
                        continue
                    target_route = routes[target_idx]
                    best_insert_cost = math.inf
                    best_insert_pos = None
                    for k in range(1, len(target_route)):
                        pred = target_route[k - 1]
                        succ = target_route[k] if k < len(target_route) else 0
                        insert_increase = distance_matrix[pred, customer] + distance_matrix[customer, succ] - distance_matrix[pred, succ]
                        if insert_increase < best_insert_cost:
                            best_insert_cost = insert_increase
                            best_insert_pos = k
                    new_target_dist = dists[target_idx] + best_insert_cost
                    new_max_candidate = max(
                        [d for idx2, d in enumerate(dists) if idx2 not in (max_route_idx, target_idx)] +
                        [new_dist_removed, new_target_dist]
                    )
                    new_total = total_dist - removal_saving + best_insert_cost
                    if (new_max_candidate < best_new_max) or (new_max_candidate == best_new_max and new_total < best_new_total):
                        best_new_max = new_max_candidate
                        best_new_total = new_total
                        best_improvement = (max_route_idx, pos, target_idx, best_insert_pos, new_dist_removed, new_target_dist, removal_saving, best_insert_cost)
            if best_improvement is None:
                break
            i_route, pos, j_route, insert_pos, new_dist_i, new_dist_j, saving, add = best_improvement
            route_i = routes[i_route]
            customer = route_i.pop(pos)
            dists[i_route] = new_dist_i
            route_j = routes[j_route]
            route_j.insert(insert_pos, customer)
            dists[j_route] = new_dist_j
            total_dist = best_new_total
            max_dist = best_new_max
            report_best_vrp(routes)

    local_search()

    # Enhanced perturbation and restart
    max_perturbations = 10
    for _ in range(max_perturbations):
        best_routes = [route[:] for route in routes]
        best_dists = dists[:]
        best_total = total_dist
        best_max = max_dist

        # Perturb: move a fraction of customers from the longest routes
        # Identify routes with max distance and other long routes
        sorted_idx = sorted(range(len(dists)), key=lambda i: dists[i], reverse=True)
        # Consider top half of routes as long routes
        num_long = max(1, len(sorted_idx) // 2)
        long_routes = sorted_idx[:num_long]
        # Collect customers from these routes
        customers_in_long = []
        for idx in long_routes:
            for c in routes[idx][1:-1]:
                customers_in_long.append((idx, c))
        if len(customers_in_long) == 0:
            break
        # Move a random fraction (30-40%) of these customers
        num_to_move = max(1, int(len(customers_in_long) * random.uniform(0.3, 0.4)))
        to_move = random.sample(customers_in_long, num_to_move)
        # Remove those customers from their routes
        new_routes = [route[:] for route in routes]
        new_dists = dists[:]
        for (r_idx, cust) in to_move:
            r = new_routes[r_idx]
            pos = r.index(cust)
            prev = r[pos-1]
            next = r[pos+1]
            new_dists[r_idx] -= distance_matrix[prev, cust] + distance_matrix[cust, next] - distance_matrix[prev, next]
            r.pop(pos)
        # Insert each moved customer into a random route (including possibly the same? but avoid)
        for (r_idx_old, cust) in to_move:
            # Choose a different route if possible
            target_idx = random.randrange(len(new_routes))
            # but ensure not the same as original? optional
            r = new_routes[target_idx]
            if len(r) > 1:
                pos = random.randint(1, len(r)-1)
            else:
                pos = 1
            pred = r[pos-1]
            succ = r[pos] if pos < len(r) else 0
            new_dists[target_idx] += distance_matrix[pred, cust] + distance_matrix[cust, succ] - distance_matrix[pred, succ]
            r.insert(pos, cust)
        routes = new_routes
        dists = new_dists
        total_dist = sum(dists)
        max_dist = max(dists)
        report_best_vrp(routes)
        # Re-apply local search
        local_search()
        # If improved, keep; else revert
        cur_max = max(dists)
        cur_total = sum(dists)
        if cur_max > best_max or (cur_max == best_max and cur_total >= best_total):
            routes = best_routes
            dists = best_dists
            total_dist = best_total
            max_dist = best_max
            report_best_vrp(routes)

    return routes