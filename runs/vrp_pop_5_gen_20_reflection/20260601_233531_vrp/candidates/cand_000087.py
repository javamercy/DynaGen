import math
import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0, 0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    def evaluate(routes):
        dists = []
        for route in routes:
            d = 0.0
            for a in range(len(route)-1):
                d += distance_matrix[route[a], route[a+1]]
            dists.append(d)
        return dists, sum(dists), max(dists)

    def report_if_better(routes, best_max, best_total):
        dists, total, maxd = evaluate(routes)
        report_best_vrp(routes)
        if maxd < best_max - 1e-12 or (abs(maxd - best_max) < 1e-12 and total < best_total - 1e-12):
            return maxd, total
        return best_max, best_total

    def two_opt_intra(route):
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
        return route

    def local_search(routes, dists, total, maxd):
        improved = True
        while improved:
            improved = False
            # Intra-route 2-opt
            for idx in range(len(routes)):
                route = routes[idx]
                if len(route) <= 3:
                    continue
                best_route = route[:]
                best_d = dists[idx]
                for i in range(1, len(route)-2):
                    for k in range(i+1, len(route)-1):
                        if k - i == 1:
                            continue
                        old_cost = distance_matrix[route[i-1], route[i]] + distance_matrix[route[k], route[k+1]]
                        new_cost = distance_matrix[route[i-1], route[k]] + distance_matrix[route[i], route[k+1]]
                        if new_cost < best_d - 1e-12:
                            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                            new_d = max(0, dists[idx] - old_cost + new_cost)
                            if new_d < best_d - 1e-12:
                                best_route = new_route
                                best_d = new_d
                if best_d < dists[idx] - 1e-12:
                    routes[idx] = best_route
                    dists[idx] = best_d
                    total = sum(dists)
                    maxd = max(dists)
                    report_best_vrp(routes)
                    improved = True
            # Inter-route: relocate, swap, 2-opt*
            for i_route in range(len(routes)):
                for j_route in range(len(routes)):
                    if i_route == j_route:
                        continue
                    # Relocate from i to j
                    for pos_i in range(1, len(routes[i_route])-1):
                        cust = routes[i_route][pos_i]
                        # Remove cost
                        prev_i = routes[i_route][pos_i-1]
                        next_i = routes[i_route][pos_i+1]
                        remove_cost = distance_matrix[prev_i, cust] + distance_matrix[cust, next_i] - distance_matrix[prev_i, next_i]
                        new_d_i = dists[i_route] - remove_cost
                        # Insert into j
                        for pos_j in range(1, len(routes[j_route])):
                            pred_j = routes[j_route][pos_j-1]
                            succ_j = routes[j_route][pos_j]
                            insert_cost = distance_matrix[pred_j, cust] + distance_matrix[cust, succ_j] - distance_matrix[pred_j, succ_j]
                            new_d_j = dists[j_route] + insert_cost
                            new_max = max(maxd, new_d_i, new_d_j) if new_d_i > maxd or new_d_j > maxd else maxd
                            # Actually we need to compute max of all routes
                            other_dists = [dists[k] for k in range(len(routes)) if k != i_route and k != j_route]
                            candidate_max = max(other_dists + [new_d_i, new_d_j])
                            candidate_total = total - remove_cost + insert_cost
                            if candidate_max < maxd - 1e-12 or (abs(candidate_max - maxd) < 1e-12 and candidate_total < total - 1e-12):
                                # Execute move
                                routes[i_route].pop(pos_i)
                                routes[j_route].insert(pos_j, cust)
                                dists[i_route] = new_d_i
                                dists[j_route] = new_d_j
                                total = candidate_total
                                maxd = candidate_max
                                report_best_vrp(routes)
                                improved = True
                                # Restart loops
                                i_route = 0
                                j_route = 0
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Swap
            for i_route in range(len(routes)):
                for j_route in range(i_route+1, len(routes)):
                    for pos_i in range(1, len(routes[i_route])-1):
                        for pos_j in range(1, len(routes[j_route])-1):
                            cust_i = routes[i_route][pos_i]
                            cust_j = routes[j_route][pos_j]
                            # Remove costs
                            prev_i = routes[i_route][pos_i-1]
                            next_i = routes[i_route][pos_i+1]
                            remove_i = distance_matrix[prev_i, cust_i] + distance_matrix[cust_i, next_i] - distance_matrix[prev_i, next_i]
                            prev_j = routes[j_route][pos_j-1]
                            next_j = routes[j_route][pos_j+1]
                            remove_j = distance_matrix[prev_j, cust_j] + distance_matrix[cust_j, next_j] - distance_matrix[prev_j, next_j]
                            # Add costs
                            add_i = distance_matrix[prev_i, cust_j] + distance_matrix[cust_j, next_i] - distance_matrix[prev_i, next_i]
                            add_j = distance_matrix[prev_j, cust_i] + distance_matrix[cust_i, next_j] - distance_matrix[prev_j, next_j]
                            new_d_i = dists[i_route] - remove_i + add_i
                            new_d_j = dists[j_route] - remove_j + add_j
                            other_dists = [dists[k] for k in range(len(routes)) if k != i_route and k != j_route]
                            candidate_max = max(other_dists + [new_d_i, new_d_j])
                            candidate_total = total - remove_i + add_i - remove_j + add_j
                            if candidate_max < maxd - 1e-12 or (abs(candidate_max - maxd) < 1e-12 and candidate_total < total - 1e-12):
                                routes[i_route][pos_i], routes[j_route][pos_j] = cust_j, cust_i
                                dists[i_route] = new_d_i
                                dists[j_route] = new_d_j
                                total = candidate_total
                                maxd = candidate_max
                                report_best_vrp(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt* inter-route
            for i_route in range(len(routes)):
                for j_route in range(i_route+1, len(routes)):
                    ri = routes[i_route]
                    rj = routes[j_route]
                    for i_idx in range(1, len(ri)-2):
                        for j_idx in range(1, len(rj)-2):
                            old_cost = distance_matrix[ri[i_idx], ri[i_idx+1]] + distance_matrix[rj[j_idx], rj[j_idx+1]]
                            new_cost = distance_matrix[ri[i_idx], rj[j_idx+1]] + distance_matrix[rj[j_idx], ri[i_idx+1]]
                            if new_cost >= old_cost - 1e-12:
                                continue
                            new_ri = ri[:i_idx+1] + rj[j_idx+1:]
                            new_rj = rj[:j_idx+1] + ri[i_idx+1:]
                            d_i = 0.0
                            for a in range(len(new_ri)-1):
                                d_i += distance_matrix[new_ri[a], new_ri[a+1]]
                            d_j = 0.0
                            for a in range(len(new_rj)-1):
                                d_j += distance_matrix[new_rj[a], new_rj[a+1]]
                            other_dists = [dists[k] for k in range(len(routes)) if k != i_route and k != j_route]
                            candidate_max = max(other_dists + [d_i, d_j])
                            candidate_total = total - dists[i_route] - dists[j_route] + d_i + d_j
                            if candidate_max < maxd - 1e-12 or (abs(candidate_max - maxd) < 1e-12 and candidate_total < total - 1e-12):
                                routes[i_route] = new_ri
                                routes[j_route] = new_rj
                                dists[i_route] = d_i
                                dists[j_route] = d_j
                                total = candidate_total
                                maxd = candidate_max
                                report_best_vrp(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes, dists, total, maxd

    def perturbation(routes, dists, total, maxd, removal_ratio):
        num_remove = max(1, int(removal_ratio * (n-1)))
        # Compute removal probabilities proportional to route distance contribution
        if maxd == 0:
            probs = [1.0/len(routes)] * len(routes)
        else:
            probs = [d / maxd for d in dists]
        total_prob = sum(probs)
        if total_prob == 0:
            probs = [1.0/len(routes)] * len(routes)
        else:
            probs = [p / total_prob for p in probs]
        removed = set()
        while len(removed) < num_remove:
            route_idx = random.choices(range(len(routes)), weights=probs, k=1)[0]
            route = routes[route_idx]
            if len(route) <= 2:
                continue
            pos = random.randint(1, len(route)-2)
            cust = route[pos]
            if cust not in removed:
                removed.add(cust)
        # Remove customers
        for cust in removed:
            for idx, route in enumerate(routes):
                if cust in route:
                    pos = route.index(cust)
                    prev = route[pos-1]
                    nxt = route[pos+1]
                    dists[idx] -= distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    route.pop(pos)
                    break
        # Reinsert using regret-2 heuristic that minimizes max distance
        removed_list = list(removed)
        random.shuffle(removed_list)
        for cust in removed_list:
            best_insertions = []
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
                if best_pos is not None:
                    best_insertions.append((idx, best_pos, best_cost))
            if not best_insertions:
                # Should not happen
                continue
            # Compute regret: for each route, compute cost; then regret = second best - best
            best_insertions.sort(key=lambda x: x[2])
            if len(best_insertions) >= 2:
                regret = best_insertions[1][2] - best_insertions[0][2]
            else:
                regret = best_insertions[0][2]
            # Find best insertion considering max distance
            best_max_after = math.inf
            best_choice = None
            for idx, pos, cost in best_insertions:
                new_dist = dists[idx] + cost
                other_dists = [dists[k] for k in range(len(routes)) if k != idx]
                candidate_max = max(other_dists + [new_dist])
                if candidate_max < best_max_after - 1e-12:
                    best_max_after = candidate_max
                    best_choice = (idx, pos, cost)
            if best_choice is None:
                # Fallback to first
                idx, pos, cost = best_insertions[0]
            else:
                idx, pos, cost = best_choice
            routes[idx].insert(pos, cust)
            dists[idx] += cost
            total += cost
            maxd = max(maxd, dists[idx])
            report_best_vrp(routes)
        return routes, dists, total, maxd

    best_routes = None
    best_max = math.inf
    best_total = math.inf
    # Multi-start with different insertion orders
    for seed in range(5):
        random.seed(seed)
        # Initial construction: greedy insertion minimizing max route distance
        routes = [[0, 0] for _ in range(truck_count)]
        dists = [0.0] * truck_count
        total = 0.0
        maxd = 0.0
        customers = list(range(1, n))
        random.shuffle(customers)
        for cust in customers:
            best_route = None
            best_pos = None
            best_new_max = math.inf
            best_new_total = math.inf
            best_add_cost = 0.0
            for idx in range(truck_count):
                route = routes[idx]
                for k in range(1, len(route)):
                    pred = route[k-1]
                    succ = route[k]
                    add_cost = distance_matrix[pred, cust] + distance_matrix[cust, succ] - distance_matrix[pred, succ]
                    new_dist = dists[idx] + add_cost
                    other_dists = [dists[j] for j in range(truck_count) if j != idx]
                    candidate_max = max(other_dists + [new_dist])
                    candidate_total = total + add_cost
                    if candidate_max < best_new_max - 1e-12 or (abs(candidate_max - best_new_max) < 1e-12 and candidate_total < best_new_total - 1e-12):
                        best_new_max = candidate_max
                        best_new_total = candidate_total
                        best_route = idx
                        best_pos = k
                        best_add_cost = add_cost
            if best_route is not None:
                routes[best_route].insert(best_pos, cust)
                dists[best_route] += best_add_cost
                total = best_new_total
                maxd = best_new_max
                report_best_vrp(routes)
        # Improve
        routes, dists, total, maxd = local_search(routes, dists, total, maxd)
        # Perturbation cycles
        for cycle in range(5):
            ratio = 0.3 - cycle * 0.05
            routes, dists, total, maxd = perturbation(routes, dists, total, maxd, ratio)
            routes, dists, total, maxd = local_search(routes, dists, total, maxd)
        if maxd < best_max - 1e-12 or (abs(maxd - best_max) < 1e-12 and total < best_total - 1e-12):
            best_max = maxd
            best_total = total
            best_routes = [route[:] for route in routes]
    if best_routes is None:
        # Fallback
        best_routes = [[0, i, 0] if i < n else [0,0] for i in range(truck_count)]
    return best_routes