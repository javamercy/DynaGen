import math
import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0, 0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    # Greedy merge construction (same as parent)
    customers = list(range(1, n))
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
                # merge i then j
                last_i = r_i[-2]
                first_j = r_j[1]
                dist_ij = dists[i] + dists[j] - distance_matrix[last_i, 0] - distance_matrix[0, first_j] + distance_matrix[last_i, first_j]
                # merge j then i
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

    total_dist = sum(dists)
    max_dist = max(dists)
    report_best_vrp(routes)

    max_iter = n * truck_count
    restart_limit = max_iter
    for restart in range(restart_limit):
        improved = True
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            # find longest route
            max_route_idx = None
            max_dist_val = -1.0
            for idx, d in enumerate(dists):
                if d > max_dist_val + 1e-12:
                    max_dist_val = d
                    max_route_idx = idx
            if max_route_idx is None or max_dist_val <= 0:
                break

            best_new_max = max_dist_val
            best_new_total = total_dist
            best_improvement = None

            # Relocate from longest route
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

            # Swap between longest route and another
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

            if best_improvement is not None:
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
                improved = True
                report_best_vrp(routes)
            iteration += 1

        # Perturbation: Ruin and Recreate
        if restart < restart_limit - 1:
            # Identify the route(s) with maximum distance
            max_dist_val = max(dists)
            longest_routes = [i for i, d in enumerate(dists) if abs(d - max_dist_val) < 1e-12]
            # Pick a random long route
            src_route_idx = random.choice(longest_routes)
            src_route = routes[src_route_idx]
            # Remove a chunk of customers from src_route (between 10% and 20% of total customers, at least 2)
            total_customers = n - 1
            num_remove = max(2, int(0.1 * total_customers) + random.randint(0, int(0.1 * total_customers)))
            num_remove = min(num_remove, len(src_route) - 2)  # cannot remove depots
            if num_remove > 0:
                # Choose random consecutive segment? Or random positions? We'll take random positions.
                indices = list(range(1, len(src_route) - 1))
                random.shuffle(indices)
                remove_indices = sorted(indices[:num_remove])
                removed_customers = [src_route[i] for i in remove_indices]
                # Remove them from src_route (iterate in reverse to avoid index issues)
                for i in reversed(remove_indices):
                    src_route.pop(i)
                # Update dists for src_route
                # Recompute its distance
                new_src_dist = 0.0
                for k in range(len(src_route) - 1):
                    new_src_dist += distance_matrix[src_route[k], src_route[k+1]]
                dists[src_route_idx] = new_src_dist
                # Now reinsert removed customers greedily into other routes to minimize max distance
                # For each customer, find best insertion (route and position) that minimizes resulting max distance
                # Break ties by total distance
                for cust in removed_customers:
                    best_new_max = math.inf
                    best_new_total = math.inf
                    best_target_route = None
                    best_insert_pos = None
                    for target_idx in range(len(routes)):
                        target_route = routes[target_idx]
                        for k in range(1, len(target_route)):
                            pred = target_route[k-1]
                            succ = target_route[k] if k < len(target_route) else 0
                            insert_cost = distance_matrix[pred, cust] + distance_matrix[cust, succ] - distance_matrix[pred, succ]
                            new_target_dist = dists[target_idx] + insert_cost
                            # compute new max distance
                            new_max = max(dists[target_idx] if i != target_idx else new_target_dist for i, d in enumerate(dists))
                            new_total = total_dist + insert_cost  # because we haven't accounted for removal? Actually total_dist already changed when we removed customers? Wait, total_dist is global and we removed customers, but we didn't adjust total_dist after removal. We'll recompute total_dist after each insertion? Simpler: maintain total_dist as sum of dists, and update it after each insertion.
                            # Instead, we can compute candidate total as sum of all dists after insertion (since we have already removed customers, but total_dist is not updated yet). We'll handle carefully.
                            # Better: before inserting, we have current dists (including src_route after removal). total_dist = sum(dists). Then we try insertion.
                            if (new_max < best_new_max - 1e-12) or (abs(new_max - best_new_max) < 1e-12 and new_target_dist - dists[target_idx] < best_new_total - total_dist):
                                best_new_max = new_max
                                best_new_total = total_dist + insert_cost
                                best_target_route = target_idx
                                best_insert_pos = k
                    # Insert the customer into the best position
                    if best_target_route is not None:
                        target_route = routes[best_target_route]
                        target_route.insert(best_insert_pos, cust)
                        dists[best_target_route] += insert_cost
                        total_dist = best_new_total
                        max_dist = max(dists)
                        report_best_vrp(routes)
                    else:
                        # Fallback: insert back into src_route
                        src_route.insert(remove_indices[0], cust)  # index may be invalid; just append
                        # Actually, better to recompute src_route distance
                        pass
            # Finally update max and total
            max_dist = max(dists)
            total_dist = sum(dists)
            report_best_vrp(routes)
        else:
            break

    return routes