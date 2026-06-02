import math
import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0, 0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    # Initial construction: assign each customer to its own route, then merge until truck_count
    routes = [[0, i, 0] for i in range(1, n)]
    dists = [2 * distance_matrix[0, i] for i in range(1, n)]
    max_dist = max(dists) if dists else 0.0
    total_dist = sum(dists)

    while len(routes) > truck_count:
        best_score = math.inf
        best_pair = None
        best_merged = None
        best_merged_dist = None
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue
                ri = routes[i]
                rj = routes[j]
                # i->j
                last_i = ri[-2]
                first_j = rj[1]
                dist_ij = dists[i] + dists[j] - distance_matrix[last_i, 0] - distance_matrix[0, first_j] + distance_matrix[last_i, first_j]
                # j->i
                last_j = rj[-2]
                first_i = ri[1]
                dist_ji = dists[i] + dists[j] - distance_matrix[last_j, 0] - distance_matrix[0, first_i] + distance_matrix[last_j, first_i]
                if dist_ij <= dist_ji:
                    new_dist = dist_ij
                    merged = ri[:-1] + rj[1:]
                else:
                    new_dist = dist_ji
                    merged = rj[:-1] + ri[1:]
                new_max = max(max_dist, new_dist)
                # Score with slight bias to max then total
                score = new_max + 1e-6 * new_dist
                if score < best_score:
                    best_score = score
                    best_pair = (i, j)
                    best_merged = merged
                    best_merged_dist = new_dist
        if best_pair is None:
            break
        i, j = best_pair
        routes[i] = best_merged
        dists[i] = best_merged_dist
        max_dist = max(max_dist, best_merged_dist)
        del routes[j]
        del dists[j]

    while len(routes) < truck_count:
        routes.append([0, 0])
        dists.append(0.0)

    total_dist = sum(dists)
    max_dist = max(dists) if dists else 0.0
    report_best_vrp(routes)

    def improve():
        nonlocal routes, dists, total_dist, max_dist
        # Intra-route 2-opt
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
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[k], route[k+1]]
                        new = distance_matrix[route[i-1], route[k]] + distance_matrix[route[i], route[k+1]]
                        if new < old - 1e-12:
                            route[i:k+1] = route[i:k+1][::-1]
                            improved = True
                            new_dist = sum(distance_matrix[route[a], route[a+1]] for a in range(len(route)-1))
                            dists[idx] = new_dist
                            total_dist = sum(dists)
                            max_dist = max(dists)
                            report_best_vrp(routes)
                            break
                    if improved:
                        break

        # Inter-route relocate, swap, 2-opt*
        max_iter = n * truck_count
        for _ in range(max_iter):
            order = sorted(range(len(routes)), key=lambda i: dists[i], reverse=True)
            improved = False
            for i_route in order:
                if improved:
                    break
                best_new_max = max_dist
                best_new_total = total_dist
                best_move = None
                route_i = routes[i_route]

                # Relocate
                for pos in range(1, len(route_i)-1):
                    cust = route_i[pos]
                    prev = route_i[pos-1]
                    nxt = route_i[pos+1]
                    saving = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_dist_i = dists[i_route] - saving
                    for j_route in range(len(routes)):
                        if j_route == i_route:
                            continue
                        route_j = routes[j_route]
                        best_insert_cost = math.inf
                        best_insert_pos = None
                        for k in range(1, len(route_j)):
                            pred = route_j[k-1]
                            succ = route_j[k]
                            cost = distance_matrix[pred, cust] + distance_matrix[cust, succ] - distance_matrix[pred, succ]
                            if cost < best_insert_cost:
                                best_insert_cost = cost
                                best_insert_pos = k
                        if best_insert_pos is None:
                            continue
                        new_dist_j = dists[j_route] + best_insert_cost
                        other_dists = [dists[idx] for idx in range(len(routes)) if idx not in (i_route, j_route)]
                        candidate_max = max(other_dists + [new_dist_i, new_dist_j])
                        candidate_total = total_dist - saving + best_insert_cost
                        if (candidate_max < best_new_max - 1e-12) or (abs(candidate_max - best_new_max) < 1e-12 and candidate_total < best_new_total - 1e-12):
                            best_new_max = candidate_max
                            best_new_total = candidate_total
                            best_move = ('relocate', i_route, pos, j_route, best_insert_pos, new_dist_i, new_dist_j)

                # Swap
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
                            candidate_max = max(other_dists + [new_dist_i, new_dist_j])
                            candidate_total = total_dist - saving_i + add_i - saving_j + add_j
                            if (candidate_max < best_new_max - 1e-12) or (abs(candidate_max - best_new_max) < 1e-12 and candidate_total < best_new_total - 1e-12):
                                best_new_max = candidate_max
                                best_new_total = candidate_total
                                best_move = ('swap', i_route, pos_i, j_route, pos_j, new_dist_i, new_dist_j)

                # 2-opt*
                for j_route in range(len(routes)):
                    if j_route == i_route:
                        continue
                    route_j = routes[j_route]
                    for i in range(1, len(route_i)-1):
                        for j in range(1, len(route_j)-1):
                            new_route_i = route_i[:i+1] + route_j[j+1:]
                            new_route_j = route_j[:j+1] + route_i[i+1:]
                            new_dist_i_cand = sum(distance_matrix[new_route_i[a], new_route_i[a+1]] for a in range(len(new_route_i)-1))
                            new_dist_j_cand = sum(distance_matrix[new_route_j[a], new_route_j[a+1]] for a in range(len(new_route_j)-1))
                            other_dists = [dists[idx] for idx in range(len(routes)) if idx not in (i_route, j_route)]
                            candidate_max = max(other_dists + [new_dist_i_cand, new_dist_j_cand])
                            candidate_total = sum(other_dists) + new_dist_i_cand + new_dist_j_cand
                            if (candidate_max < best_new_max - 1e-12) or (abs(candidate_max - best_new_max) < 1e-12 and candidate_total < best_new_total - 1e-12):
                                best_new_max = candidate_max
                                best_new_total = candidate_total
                                best_move = ('2opt*', i_route, j_route, i, j, new_route_i, new_route_j, new_dist_i_cand, new_dist_j_cand)

                if best_move is not None:
                    if best_move[0] == 'relocate':
                        _, i_route, pos, j_route, insert_pos, new_dist_i, new_dist_j = best_move
                        cust = routes[i_route].pop(pos)
                        dists[i_route] = new_dist_i
                        routes[j_route].insert(insert_pos, cust)
                        dists[j_route] = new_dist_j
                    elif best_move[0] == 'swap':
                        _, i_route, pos_i, j_route, pos_j, new_dist_i, new_dist_j = best_move
                        cust_i = routes[i_route][pos_i]
                        cust_j = routes[j_route][pos_j]
                        routes[i_route][pos_i] = cust_j
                        routes[j_route][pos_j] = cust_i
                        dists[i_route] = new_dist_i
                        dists[j_route] = new_dist_j
                    else:  # 2opt*
                        _, i_route, j_route, i, j, new_route_i, new_route_j, new_dist_i, new_dist_j = best_move
                        routes[i_route] = new_route_i
                        routes[j_route] = new_route_j
                        dists[i_route] = new_dist_i
                        dists[j_route] = new_dist_j
                    total_dist = best_new_total
                    max_dist = best_new_max
                    report_best_vrp(routes)
                    improved = True
                    break
            if not improved:
                break

    def perturbation(remove_ratio):
        nonlocal routes, dists, total_dist, max_dist
        n_removed = max(1, min(int(remove_ratio * (n-1)), n-1))
        removed_set = set()
        while len(removed_set) < n_removed:
            route_idx = random.randrange(len(routes))
            route = routes[route_idx]
            if len(route) <= 2:
                continue
            pos = random.randint(1, len(route)-2)
            cust = route[pos]
            if cust not in removed_set:
                removed_set.add(cust)
        for cust in removed_set:
            for idx, route in enumerate(routes):
                if cust in route:
                    pos = route.index(cust)
                    prev = route[pos-1]
                    nxt = route[pos+1]
                    dists[idx] -= distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    route.pop(pos)
                    break
        removed_list = list(removed_set)
        random.shuffle(removed_list)
        penalty_coeff = 1.0
        max_d = max_dist if max_dist > 0 else 1.0
        for cust in removed_list:
            best_costs = []
            best_positions = []
            for idx, route in enumerate(routes):
                best_cost = math.inf
                best_pos = None
                for k in range(1, len(route)):
                    pred = route[k-1]
                    succ = route[k]
                    delta = distance_matrix[pred, cust] + distance_matrix[cust, succ] - distance_matrix[pred, succ]
                    penalty = penalty_coeff * (dists[idx] / max_d)
                    cost = delta + penalty
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = k
                if best_pos is not None:
                    real_delta = distance_matrix[route[best_pos-1], cust] + distance_matrix[cust, route[best_pos]] - distance_matrix[route[best_pos-1], route[best_pos]]
                    best_costs.append(real_delta)
                    best_positions.append((idx, best_pos))
            if not best_costs:
                continue
            # Regret: difference between best and second best
            if len(best_costs) >= 2:
                sorted_costs = sorted(best_costs)
                regret = sorted_costs[1] - sorted_costs[0]
            else:
                regret = best_costs[0]
            # Choose best insertion considering max distance
            best_max_after = math.inf
            best_route_idx = None
            best_insert_pos = None
            for (idx, pos), delta in zip(best_positions, best_costs):
                new_dist = dists[idx] + delta
                other_dists = [dists[j] for j in range(len(routes)) if j != idx]
                candidate_max = max(other_dists + [new_dist])
                if candidate_max < best_max_after - 1e-12:
                    best_max_after = candidate_max
                    best_route_idx = idx
                    best_insert_pos = pos
            if best_route_idx is not None:
                route = routes[best_route_idx]
                route.insert(best_insert_pos, cust)
                actual_delta = distance_matrix[route[best_insert_pos-1], cust] + distance_matrix[cust, route[best_insert_pos+1]] - distance_matrix[route[best_insert_pos-1], route[best_insert_pos+1]]
                dists[best_route_idx] += actual_delta
                total_dist = sum(dists)
                max_dist = max(dists)
                report_best_vrp(routes)

    # Main loop: improvement + perturbation cycles
    improve()
    for cycle in range(5):
        remove_ratio = 0.3 - cycle * 0.05
        if n > 2:
            perturbation(remove_ratio)
            improve()

    # Final best solution is already tracked via report_best_vrp
    # Return the last routes (which should be the best)
    return routes