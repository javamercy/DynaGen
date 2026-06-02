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
    # Construction: each customer as separate route
    routes = [[0, c, 0] for c in customers]
    dists = [2 * distance_matrix[0, c] for c in customers]

    def route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # Merge until truck_count routes
    while len(routes) > truck_count:
        best_new_max = math.inf
        best_new_total = math.inf
        best_pair = None
        best_orientation = None
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                ri = routes[i]
                rj = routes[j]
                # orientation i then j
                dist_ij = dists[i] + dists[j] - distance_matrix[ri[-2], 0] - distance_matrix[0, rj[1]] + distance_matrix[ri[-2], rj[1]]
                # orientation j then i
                dist_ji = dists[i] + dists[j] - distance_matrix[rj[-2], 0] - distance_matrix[0, ri[1]] + distance_matrix[rj[-2], ri[1]]
                new_max_ij = max(dists[i], dists[j], dist_ij)  # since other routes unchanged
                new_max_ji = max(dists[i], dists[j], dist_ji)
                # tie-break by total
                total_without = sum(dists) - dists[i] - dists[j]
                if new_max_ij < best_new_max or (new_max_ij == best_new_max and total_without + dist_ij < best_new_total):
                    best_new_max = new_max_ij
                    best_new_total = total_without + dist_ij
                    best_pair = (i, j)
                    best_orientation = 'ij'
                if new_max_ji < best_new_max or (new_max_ji == best_new_max and total_without + dist_ji < best_new_total):
                    best_new_max = new_max_ji
                    best_new_total = total_without + dist_ji
                    best_pair = (i, j)
                    best_orientation = 'ji'
        if best_pair is None:
            break
        i, j = best_pair
        if best_orientation == 'ij':
            merged = routes[i][:-1] + routes[j][1:]
            dist_merged = best_new_total - (sum(dists) - dists[i] - dists[j])
        else:
            merged = routes[j][:-1] + routes[i][1:]
            dist_merged = best_new_total - (sum(dists) - dists[i] - dists[j])
        routes[i] = merged
        dists[i] = dist_merged
        del routes[j]
        del dists[j]

    # Add empty routes if under truck_count
    while len(routes) < truck_count:
        routes.append([0, 0])
        dists.append(0.0)

    total_dist = sum(dists)
    max_dist = max(dists)
    report_best_vrp(routes)

    # Local search: relocate customers from longest route
    def local_search():
        nonlocal routes, dists, total_dist, max_dist
        max_iter = n * truck_count
        improved = True
        while improved:
            improved = False
            for _ in range(max_iter):
                # Find longest route
                max_idx = max(range(len(routes)), key=lambda idx: dists[idx])
                if dists[max_idx] == 0:
                    break
                route = routes[max_idx]
                best_improvement = None
                best_new_max = max_dist
                best_new_total = total_dist
                for pos in range(1, len(route)-1):
                    customer = route[pos]
                    prev = route[pos-1]
                    next_ = route[pos+1]
                    removal_saving = distance_matrix[prev, customer] + distance_matrix[customer, next_] - distance_matrix[prev, next_]
                    new_dist_i = dists[max_idx] - removal_saving
                    for target_idx in range(len(routes)):
                        if target_idx == max_idx:
                            continue
                        target_route = routes[target_idx]
                        # Find best insertion position in target route
                        best_insert_cost = math.inf
                        best_insert_pos = None
                        for k in range(1, len(target_route)):
                            pred = target_route[k-1]
                            succ = target_route[k] if k < len(target_route) else 0
                            cost = distance_matrix[pred, customer] + distance_matrix[customer, succ] - distance_matrix[pred, succ]
                            if cost < best_insert_cost:
                                best_insert_cost = cost
                                best_insert_pos = k
                        new_dist_j = dists[target_idx] + best_insert_cost
                        new_max_candidate = max(
                            [d for idx_, d in enumerate(dists) if idx_ not in (max_idx, target_idx)] +
                            [new_dist_i, new_dist_j]
                        )
                        new_total = total_dist - removal_saving + best_insert_cost
                        if (new_max_candidate < best_new_max) or (new_max_candidate == best_new_max and new_total < best_new_total):
                            best_new_max = new_max_candidate
                            best_new_total = new_total
                            best_improvement = (max_idx, pos, target_idx, best_insert_pos, new_dist_i, new_dist_j)
                if best_improvement is None:
                    break
                i_route, pos, j_route, ins_pos, new_dist_i, new_dist_j = best_improvement
                cust = routes[i_route].pop(pos)
                dists[i_route] = new_dist_i
                routes[j_route].insert(ins_pos, cust)
                dists[j_route] = new_dist_j
                total_dist = best_new_total
                max_dist = best_new_max
                report_best_vrp(routes)
                improved = True

    local_search()

    # Perturbation and restart
    best_routes = [route[:] for route in routes]
    best_dists = dists[:]
    best_total = total_dist
    best_max = max_dist

    for _ in range(5):
        # Randomly destroy a fraction of customers
        all_customers = []
        for route in routes:
            all_customers.extend(route[1:-1])
        if len(all_customers) == 0:
            break
        num_to_move = max(1, len(all_customers) // 5)  # 20% but we'll use 30% for stronger perturbation
        # Actually set to 30%
        num_to_move = max(1, len(all_customers) * 3 // 10)
        to_move = random.sample(all_customers, num_to_move)
        # Remove them from routes
        new_routes = [route[:] for route in routes]
        new_dists = dists[:]
        for cust in to_move:
            for r_idx, r in enumerate(new_routes):
                if cust in r:
                    pos = r.index(cust)
                    prev = r[pos-1]
                    next_ = r[pos+1]
                    new_dists[r_idx] -= distance_matrix[prev, cust] + distance_matrix[cust, next_] - distance_matrix[prev, next_]
                    r.pop(pos)
                    break
        # Greedy reinsert to minimize max distance
        random.shuffle(to_move)  # add randomness
        for cust in to_move:
            best_insert = None
            best_new_max_after = math.inf
            best_new_total_after = math.inf
            for r_idx in range(len(new_routes)):
                route = new_routes[r_idx]
                # evaluate insertion at each position
                for k in range(1, len(route)):
                    pred = route[k-1]
                    succ = route[k] if k < len(route) else 0
                    cost = distance_matrix[pred, cust] + distance_matrix[cust, succ] - distance_matrix[pred, succ]
                    new_dist = new_dists[r_idx] + cost
                    # compute new max and total
                    other_dists = [new_dists[idx] for idx in range(len(new_routes)) if idx != r_idx]
                    new_max_candidate = max(max(other_dists + [new_dist]), new_dist if not other_dists else max(other_dists + [new_dist]))
                    new_total = sum(new_dists) - new_dists[r_idx] + new_dist + cost
                    if (new_max_candidate < best_new_max_after) or (new_max_candidate == best_new_max_after and new_total < best_new_total_after):
                        best_new_max_after = new_max_candidate
                        best_new_total_after = new_total
                        best_insert = (r_idx, k, new_dist)
            if best_insert is None:
                # fallback: insert at end of an arbitrary route
                r_idx = 0
                k = len(new_routes[0])
                cost = distance_matrix[new_routes[0][-2], cust] + distance_matrix[cust, 0] - distance_matrix[new_routes[0][-2], 0]
                new_dists[r_idx] += cost
                new_routes[r_idx].insert(k, cust)
            else:
                r_idx, k, new_dist = best_insert
                new_dists[r_idx] = new_dist
                new_routes[r_idx].insert(k, cust)
        routes = new_routes
        dists = new_dists
        total_dist = sum(dists)
        max_dist = max(dists)
        report_best_vrp(routes)
        # Local search
        local_search()
        # Check if improved over best
        cur_max = max(dists)
        cur_total = sum(dists)
        if cur_max < best_max or (cur_max == best_max and cur_total < best_total):
            best_routes = [route[:] for route in routes]
            best_dists = dists[:]
            best_total = cur_total
            best_max = cur_max
        else:
            # revert
            routes = [route[:] for route in best_routes]
            dists = best_dists[:]
            total_dist = best_total
            max_dist = best_max
            report_best_vrp(routes)
    return routes