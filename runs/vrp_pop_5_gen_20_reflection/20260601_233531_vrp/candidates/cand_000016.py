import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0, 0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    customers = list(range(1, n))
    # each customer in its own route
    routes = [[0, c, 0] for c in customers]
    dists = [2 * distance_matrix[0, c] for c in customers]

    # merge routes until truck_count
    while len(routes) > truck_count:
        best_new_max = math.inf
        best_new_total = math.inf
        best_pair = None
        best_merged_route = None
        current_max = max(dists)
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue
                r1 = routes[i]
                r2 = routes[j]
                # try merge orientation: r1 then r2
                merged1 = r1[:-1] + r2[1:]  # remove depot at end of r1 and start of r2
                dist1 = 0
                for k in range(len(merged1)-1):
                    dist1 += distance_matrix[merged1[k], merged1[k+1]]
                new_max1 = max(current_max, dist1, dists[i] + dists[j] - dist1)  # careful: actually after merge, only one route remains, so max is max of all other distances and merged distance
                # all other routes unchanged, so new max = max(max(dists without i and j), dist1)
                # but we need to compute max of remaining + merged
                # let's compute properly:
                other_dists = [dists[k] for k in range(len(dists)) if k not in (i,j)]
                new_max1 = max(max(other_dists, default=0), dist1)
                new_total1 = sum(other_dists) + dist1

                # try merge orientation: r2 then r1
                merged2 = r2[:-1] + r1[1:]
                dist2 = 0
                for k in range(len(merged2)-1):
                    dist2 += distance_matrix[merged2[k], merged2[k+1]]
                new_max2 = max(max(other_dists, default=0), dist2)
                new_total2 = sum(other_dists) + dist2

                if new_max1 < best_new_max or (new_max1 == best_new_max and new_total1 < best_new_total):
                    best_new_max = new_max1
                    best_new_total = new_total1
                    best_pair = (i, j)
                    best_merged_route = merged1
                    best_new_dist = dist1
                if new_max2 < best_new_max or (new_max2 == best_new_max and new_total2 < best_new_total):
                    best_new_max = new_max2
                    best_new_total = new_total2
                    best_pair = (i, j)
                    best_merged_route = merged2
                    best_new_dist = dist2

        if best_pair is None:
            break
        i, j = best_pair
        # We need to order i and j correctly: since we considered both orientations, we need to know which route to keep.
        # We'll remove route i and j and add merged route. Since we tracked best_pair as (i,j) but the merged route could be either.
        # To avoid confusion, we'll deduce: if best_merged_route == routes[i][:-1] + routes[j][1:], then keep i as merged and delete j; else keep j as merged and delete i.
        if best_merged_route == routes[i][:-1] + routes[j][1:]:
            routes[i] = best_merged_route
            dists[i] = best_new_dist
            del routes[j]
            del dists[j]
        else:
            routes[j] = best_merged_route
            dists[j] = best_new_dist
            del routes[i]
            del dists[i]

    # pad with empty routes if needed
    while len(routes) < truck_count:
        routes.append([0, 0])
        dists.append(0.0)

    total_dist = sum(dists)
    report_best_vrp(routes)

    # local search: relocate and swap moves from longest route
    max_iter = n * truck_count
    for _ in range(max_iter):
        # find longest route index
        max_dist = -1
        max_idx = -1
        for idx, d in enumerate(dists):
            if d > max_dist:
                max_dist = d
                max_idx = idx
        if max_dist == 0:
            break
        best_improvement = None
        best_new_max = max_dist
        best_new_total = total_dist

        # consider all customers in longest route
        route = routes[max_idx]
        for pos in range(1, len(route)-1):
            cust = route[pos]
            prev = route[pos-1]
            nxt = route[pos+1]
            remove_saving = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
            new_dist_removed = dists[max_idx] - remove_saving

            # try relocate to another route
            for target_idx in range(len(routes)):
                if target_idx == max_idx:
                    continue
                target_route = routes[target_idx]
                best_insert_cost = math.inf
                best_insert_pos = None
                for k in range(1, len(target_route)):
                    pred = target_route[k-1]
                    succ = target_route[k] if k < len(target_route) else 0
                    insert_increase = distance_matrix[pred, cust] + distance_matrix[cust, succ] - distance_matrix[pred, succ]
                    if insert_increase < best_insert_cost:
                        best_insert_cost = insert_increase
                        best_insert_pos = k
                new_target_dist = dists[target_idx] + best_insert_cost
                # compute new max
                other_dists = [dists[k] for k in range(len(dists)) if k not in (max_idx, target_idx)]
                new_max_candidate = max(other_dists + [new_dist_removed, new_target_dist])
                new_total = total_dist - remove_saving + best_insert_cost
                if (new_max_candidate < best_new_max) or (new_max_candidate == best_new_max and new_total < best_new_total):
                    best_new_max = new_max_candidate
                    best_new_total = new_total
                    best_improvement = ('relocate', max_idx, pos, target_idx, best_insert_pos, new_dist_removed, new_target_dist)

            # try swap with a customer from another route
            for target_idx in range(len(routes)):
                if target_idx == max_idx:
                    continue
                target_route = routes[target_idx]
                for swap_pos in range(1, len(target_route)-1):
                    swap_cust = target_route[swap_pos]
                    # compute new max for both routes after swapping
                    # remove cust from max route
                    # compute target route after removing swap_cust and inserting cust
                    # and max route after removing cust and inserting swap_cust
                    # compute costs
                    # remove cust from max route
                    prev_c = route[pos-1]
                    nxt_c = route[pos+1]
                    removal_saving_cust = distance_matrix[prev_c, cust] + distance_matrix[cust, nxt_c] - distance_matrix[prev_c, nxt_c]
                    new_dist_max = dists[max_idx] - removal_saving_cust
                    # after removal, insert swap_cust into max route at best position? We'll try all insert positions for swap_cust in max route (excluding pos? Actually we are swapping, so we remove cust from max and insert swap_cust, and remove swap_cust from target and insert cust). We need to consider all insertion positions for swap_cust in the max route (after removal of cust). But for simplicity, we can also try all insertion positions. However, to keep it bounded, we can consider inserting swap_cust at the same position or near. Instead, we'll compute the best insertion for swap_cust into max route (after removal of cust) and best insertion for cust into target route (after removal of swap_cust). This is a standard swap move.
                    # We'll compute the best insertion positions for both.
                    # First, for target route after removing swap_cust:
                    prev_s = target_route[swap_pos-1]
                    nxt_s = target_route[swap_pos+1] if swap_pos+1 < len(target_route) else 0
                    removal_saving_swap = distance_matrix[prev_s, swap_cust] + distance_matrix[swap_cust, nxt_s] - distance_matrix[prev_s, nxt_s]
                    new_dist_target = dists[target_idx] - removal_saving_swap

                    # Now insert cust into target route (after removal of swap_cust)
                    best_insert_cust = math.inf
                    best_pos_cust = None
                    # The new target route after removal has length len(target_route)-1 (removed one customer). Insert positions: between 1 and len-1
                    new_target_route_len = len(target_route) - 1  # after removal
                    for k in range(1, new_target_route_len):  # k is index between nodes, same as before
                        # need the actual node before and after k in the new route
                        # The new route after removal is target_route[:swap_pos] + target_route[swap_pos+1:]
                        # We'll create it temporarily, but to avoid overhead, we can compute directly
                        # Let's compute predecessor and successor for insertion at position k
                        if k <= swap_pos:
                            pred = target_route[k-1]
                            succ = target_route[k] if k < swap_pos else target_route[swap_pos+1] if k == swap_pos else target_route[k]  # careful
                        else:
                            pred = target_route[k] if k > swap_pos else target_route[k-1]
                            succ = target_route[k+1] if k+1 < len(target_route) else 0
                        # Actually simpler: we can build the new route and compute distance for each insertion, but that may be heavy. Since n is small (100), we can just create the new route for each swap evaluation. It's okay because loops are bounded.
                        # We'll do the simpler: Create new route for max after removal and insertion, and for target similarly, and compute distances.
                    # To keep code manageable, we'll compute efficiently but for clarity we can do straightforward loops.
                    # For now, we'll skip swap in this thought and focus on relocate? But reflection demands swaps.
                    # I'll implement swap by enumerating all insertion positions for both routes.
                    # Prepare new route for max without cust
                    new_max_route = route[:pos] + route[pos+1:]
                    # Prepare new route for target without swap_cust
                    new_target_route = target_route[:swap_pos] + target_route[swap_pos+1:]
                    # For each insertion of swap_cust into new_max_route
                    best_new_max_dist = math.inf
                    best_new_target_dist = math.inf
                    best_insert_pos_max = None
                    best_insert_pos_target = None
                    for pm in range(1, len(new_max_route)):  # insert swap_cust at position pm (between nodes)
                        # compute cost
                        pred_m = new_max_route[pm-1]
                        succ_m = new_max_route[pm] if pm < len(new_max_route) else 0
                        insert_cost_m = distance_matrix[pred_m, swap_cust] + distance_matrix[swap_cust, succ_m] - distance_matrix[pred_m, succ_m]
                        new_dist_max_candidate = new_dist_max + insert_cost_m
                        for pt in range(1, len(new_target_route)):
                            pred_t = new_target_route[pt-1]
                            succ_t = new_target_route[pt] if pt < len(new_target_route) else 0
                            insert_cost_t = distance_matrix[pred_t, cust] + distance_matrix[cust, succ_t] - distance_matrix[pred_t, succ_t]
                            new_dist_target_candidate = new_dist_target + insert_cost_t
                            max_candidate = max(dists[k] for k in range(len(dists)) if k not in (max_idx, target_idx)) if len(dists) > 2 else 0
                            max_candidate = max(max_candidate, new_dist_max_candidate, new_dist_target_candidate)
                            total_candidate = total_dist - removal_saving_cust - removal_saving_swap + insert_cost_m + insert_cost_t
                            if (max_candidate < best_new_max) or (max_candidate == best_new_max and total_candidate < best_new_total):
                                best_new_max = max_candidate
                                best_new_total = total_candidate
                                best_improvement = ('swap', max_idx, pos, target_idx, swap_pos, pm, pt)

        if best_improvement is None:
            break
        # apply the best improvement
        if best_improvement[0] == 'relocate':
            _, i_route, pos, j_route, ins_pos, new_dist_i, new_dist_j = best_improvement
            route_i = routes[i_route]
            cust = route_i.pop(pos)
            dists[i_route] = new_dist_i
            route_j = routes[j_route]
            route_j.insert(ins_pos, cust)
            dists[j_route] = new_dist_j
            total_dist = best_new_total
        else: # swap
            _, i_route, pos_i, j_route, pos_j, ins_i, ins_j = best_improvement
            route_i = routes[i_route]
            route_j = routes[j_route]
            cust_i = route_i.pop(pos_i)
            cust_j = route_j.pop(pos_j)
            dists[i_route] = dists[i_route] - distance_matrix[route_i[pos_i-1] if pos_i>0 else 0, cust_i] - distance_matrix[cust_i, route_i[pos_i] if pos_i<len(route_i) else 0] + distance_matrix[route_i[pos_i-1] if pos_i>0 else 0, route_i[pos_i] if pos_i<len(route_i) else 0]  # but we have computed insert costs, easier to set dists directly from computed new dists
            # Actually we have new_dist_i and new_dist_j from the best improvement? We didn't store them. We need to compute them.
            # We'll redo the computation here for simplicity.
            # Let's recompute the new distances for the selected move.
            # Since we only select one move, we can recompute.
            # For relocate we already stored new_dist_i and new_dist_j.
            # For swap, we need to compute new distances after applying the swap.
            # We'll compute them inline.
            # To avoid complexity, we can compute new routes and distances after applying the swap.
            # We'll just do that.
            # First remove cust_i from route_i and cust_j from route_j
            route_i_removed = route_i[:pos_i] + route_i[pos_i+1:]
            route_j_removed = route_j[:pos_j] + route_j[pos_j+1:]
            # Insert cust_j into route_i_removed at ins_i
            route_i_new = route_i_removed[:ins_i] + [cust_j] + route_i_removed[ins_i:]
            # Insert cust_i into route_j_removed at ins_j
            route_j_new = route_j_removed[:ins_j] + [cust_i] + route_j_removed[ins_j:]
            # Update routes and distances
            routes[i_route] = route_i_new
            routes[j_route] = route_j_new
            # Compute new distances
            dist_i_new = 0
            for k in range(len(route_i_new)-1):
                dist_i_new += distance_matrix[route_i_new[k], route_i_new[k+1]]
            dists[i_route] = dist_i_new
            dist_j_new = 0
            for k in range(len(route_j_new)-1):
                dist_j_new += distance_matrix[route_j_new[k], route_j_new[k+1]]
            dists[j_route] = dist_j_new
            # Update total_dist
            total_dist = best_new_total  # better to compute from dists sum
            total_dist = sum(dists)
            # Don't forget to call report_best_vrp
        report_best_vrp(routes)

    return routes