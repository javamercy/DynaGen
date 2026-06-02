import math
import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0, 0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    def improve(routes, dists, total_dist, max_dist):
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
                                new_dist = 0.0
                                for a in range(len(route)-1):
                                    new_dist += distance_matrix[route[a], route[a+1]]
                                dists[idx] = new_dist
                                total_dist = sum(dists)
                                max_dist = max(dists)
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
        # Inter-route best-improvement relocate/swap
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
                    report_best_vrp(routes)
                    improved = True
                    break
            if not improved:
                break
        # Inter-route 2-opt* moves
        max_iter_2opt = n * truck_count
        for iteration in range(max_iter_2opt):
            improved = False
            for i_route in range(len(routes)):
                if improved:
                    break
                for j_route in range(i_route+1, len(routes)):
                    if improved:
                        break
                    route_i = routes[i_route]
                    route_j = routes[j_route]
                    if len(route_i) <= 2 or len(route_j) <= 2:
                        continue
                    # Consider all splits
                    for pos_i in range(1, len(route_i)-1):
                        for pos_j in range(1, len(route_j)-1):
                            # Compute cost of current edges
                            # Edge from route_i[pos_i-1] to route_i[pos_i] and from route_i[pos_i] to route_i[pos_i+1]
                            # Similarly for route_j
                            # In 2-opt*, we swap the tails after pos_i and pos_j
                            # New route_i: first part of route_i up to pos_i, then reversed? Actually standard 2-opt*: keep orientation, just swap the end segments.
                            # So new route_i: route_i[0:pos_i+1] + route_j[pos_j+1:] (assuming direction preserved)
                            # But need to ensure start and end at depot. Actually we are using sequence [0, ..., 0]. So we can concatenate.
                            # For simplicity, we can treat the routes as lists without depot endpoints? We have depots at index 0 and -1.
                            # So route_i[0]=0, route_i[-1]=0. We'll only consider internal positions 1..len-2.
                            # New route_i: route_i[0:pos_i+1] + route_j[pos_j+1:] (but then the last element of route_j[pos_j+1:] is 0, so we need to ensure it ends with 0.
                            # Actually route_j ends with 0, so route_j[pos_j+1:] starts from the element after pos_j and includes 0 at the end. So concatenation yields route_i ending with 0.
                            # However we must also consider swapping the order (i.e., also the other combination).
                            # We'll compute both options.
                            # Option 1: new_i = route_i[:pos_i+1] + route_j[pos_j+1:]
                            # Option 2: new_i = route_i[:pos_i+1] + route_j[1:pos_j+1][::-1]? That would reverse direction, but since distances are symmetric it doesn't matter? Actually reversing order of customers changes sequence but cost same? Not exactly because edges change. We'll consider both possible connections.
                            # Standard 2-opt* for VRP normally maintains direction; we'll just do the simple swap of tails (option 1) and also the reverse (option 2).
                            # But to keep it simple, we'll compute cost for swap of tails (i.e., new_i = route_i[:pos_i+1] + route_j[pos_j+1:]; new_j = route_j[:pos_j+1] + route_i[pos_i+1:]).
                            # That's the usual 2-opt*.
                            new_route_i = route_i[:pos_i+1] + route_j[pos_j+1:]
                            new_route_j = route_j[:pos_j+1] + route_i[pos_i+1:]
                            # Compute distances
                            new_dist_i = 0.0
                            for a in range(len(new_route_i)-1):
                                new_dist_i += distance_matrix[new_route_i[a], new_route_i[a+1]]
                            new_dist_j = 0.0
                            for a in range(len(new_route_j)-1):
                                new_dist_j += distance_matrix[new_route_j[a], new_route_j[a+1]]
                            other_dists = [dists[k] for k in range(len(routes)) if k not in (i_route, j_route)]
                            combined = other_dists + [new_dist_i, new_dist_j]
                            candidate_max = max(combined)
                            candidate_total = total_dist - dists[i_route] - dists[j_route] + new_dist_i + new_dist_j
                            if (candidate_max < max_dist - 1e-12) or (abs(candidate_max - max_dist) < 1e-12 and candidate_total < total_dist - 1e-12):
                                # Apply move
                                routes[i_route] = new_route_i
                                routes[j_route] = new_route_j
                                dists[i_route] = new_dist_i
                                dists[j_route] = new_dist_j
                                total_dist = candidate_total
                                max_dist = candidate_max
                                report_best_vrp(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        return routes, dists, total_dist, max_dist

    def perturbation(routes, dists, total_dist, max_dist, remove_ratio):
        n_removed = max(1, min(int(remove_ratio * (n-1)), n-1))
        # Compute route weights for removal (higher distance => higher probability)
        max_d = max_dist
        weights = []
        for dist in dists:
            if dist > 0:
                weights.append(dist / max_d)
            else:
                weights.append(0.0)
        # Normalize weights to probabilities
        total_weight = sum(weights)
        if total_weight == 0:
            probs = [1.0/len(weights)] * len(weights)
        else:
            probs = [w / total_weight for w in weights]
        # Select customers to remove
        customers = list(range(1, n))
        removed_set = set()
        while len(removed_set) < n_removed:
            # Pick a route based on probs
            route_idx = random.choices(range(len(routes)), weights=probs, k=1)[0]
            route = routes[route_idx]
            if len(route) <= 2:
                continue
            # Random customer from that route (excluding depot)
            pos = random.randint(1, len(route)-2)
            cust = route[pos]
            if cust not in removed_set:
                removed_set.add(cust)
        # Remove selected customers
        for cust in removed_set:
            for idx, route in enumerate(routes):
                if cust in route:
                    pos = route.index(cust)
                    prev = route[pos-1]
                    nxt = route[pos+1]
                    dists[idx] -= distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    route.pop(pos)
                    break
        # Reinsert using regret-2 heuristic
        removed_list = list(removed_set)
        random.shuffle(removed_list)
        for cust in removed_list:
            best_costs = []
            best_positions = []
            for idx, route in enumerate(routes):
                best_cost = math.inf
                best_pos = None
                for k in range(1, len(route)):
                    pred = route[k-1]
                    succ = route[k]
                    cost = distance_matrix[pred, cust] + distance_matrix[cust, succ] - distance_matrix[pred, succ]
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = k
                if best_cost < math.inf:
                    best_costs.append(best_cost)
                    best_positions.append((idx, best_pos))
            if len(best_costs) == 0:
                # Only one route? treat as best insertion
                # Actually we should always have at least one route with capacity
                # But to be safe, assign to first route
                route_idx = 0
                insert_pos = 1
                best_costs = [distance_matrix[0, cust] + distance_matrix[cust, 0]]
                best_positions = [(0, 1)]
            # Compute regret (difference between best and second best)
            if len(best_costs) >= 2:
                sorted_costs = sorted(best_costs)
                regret = sorted_costs[1] - sorted_costs[0]
            else:
                regret = best_costs[0]
            # Choose customer with highest regret (or if tie, random)
            # Since we process one at a time, we compute regret for current customer and insert immediately
            # But to mimic regret-2, we should compute for all pending customers?
            # For simplicity, we insert each customer with best insertion minimizing max distance, but use regret to choose order
            # We'll compute regret for this customer and then insert at best position
            # Choose route and position that minimizes candidate max
            best_max_after = math.inf
            best_route_idx = None
            best_insert_pos = None
            for (idx, pos), cost in zip(best_positions, best_costs):
                new_dist = dists[idx] + cost
                other_dists = [dists[j] for j in range(len(routes)) if j != idx]
                candidate_max = max(other_dists + [new_dist])
                if candidate_max < best_max_after - 1e-12:
                    best_max_after = candidate_max
                    best_route_idx = idx
                    best_insert_pos = pos
            if best_route_idx is not None:
                route = routes[best_route_idx]
                route.insert(best_insert_pos, cust)
                dists[best_route_idx] += distance_matrix[route[best_insert_pos-1], cust] + distance_matrix[cust, route[best_insert_pos+1]] - distance_matrix[route[best_insert_pos-1], route[best_insert_pos+1]]
                total_dist = sum(dists)
                max_dist = max(dists)
                report_best_vrp(routes)
        return routes, dists, total_dist, max_dist

    best_routes = None
    best_max = math.inf
    best_total = math.inf
    alphas = [1.0, 0.75, 0.5, 0.25, 0.0]
    for alpha in alphas:
        # Construction: each customer as a single route
        routes = [[0, i, 0] for i in range(1, n)]
        dists = [2 * distance_matrix[0, i] for i in range(1, n)]
        current_max = max(dists) if dists else 0.0
        while len(routes) > truck_count:
            best_score = math.inf
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
                    score = alpha * new_max + (1 - alpha) * new_dist
                    if score < best_score - 1e-12:
                        best_score = score
                        best_pair = (i, j)
                        best_merged_route = merged
                        best_merged_dist = new_dist
            if best_pair is None:
                break
            i, j = best_pair
            routes[i] = best_merged_route
            dists[i] = best_merged_dist
            current_max = max(current_max, best_merged_dist)
            del routes[j]
            del dists[j]
        while len(routes) < truck_count:
            routes.append([0, 0])
            dists.append(0.0)
        total_dist = sum(dists)
        max_dist = max(dists) if dists else 0.0
        report_best_vrp(routes)
        routes, dists, total_dist, max_dist = improve(routes, dists, total_dist, max_dist)
        # Perturbation cycles with decreasing removal ratio
        for cycle in range(5):
            remove_ratio = 0.3 - cycle * 0.05  # from 0.3 down to 0.1
            if n > 2:
                routes, dists, total_dist, max_dist = perturbation(routes, dists, total_dist, max_dist, remove_ratio)
                routes, dists, total_dist, max_dist = improve(routes, dists, total_dist, max_dist)
        if max_dist < best_max - 1e-12 or (abs(max_dist - best_max) < 1e-12 and total_dist < best_total - 1e-12):
            best_max = max_dist
            best_total = total_dist
            best_routes = [route[:] for route in routes]
    return best_routes