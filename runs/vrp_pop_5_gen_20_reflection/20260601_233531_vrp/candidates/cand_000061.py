import math
import random
import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n == 1:
        routes = [[0, 0] for _ in range(truck_count)]
        report_best_vrp(routes)
        return routes

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_dists(routes):
        return [route_distance(r) for r in routes]

    def improvement(routes, dists):
        total_dist = sum(dists)
        max_dist = max(dists)
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
                        old_cost = distance_matrix[route[i-1], route[i]] + distance_matrix[route[k], route[k+1]]
                        new_cost = distance_matrix[route[i-1], route[k]] + distance_matrix[route[i], route[k+1]]
                        if new_cost < old_cost - 1e-12:
                            route[i:k+1] = route[i:k+1][::-1]
                            new_d = route_distance(route)
                            dists[idx] = new_d
                            total_dist = sum(dists)
                            max_dist = max(dists)
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
        # Inter-route best-improvement: relocate, swap, 2-opt*
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
                # 2-opt* inter-route moves
                for j_route in range(len(routes)):
                    if j_route == i_route:
                        continue
                    route_j = routes[j_route]
                    for i_idx in range(1, len(route_i) - 1):
                        for j_idx in range(1, len(route_j) - 1):
                            old_cost = distance_matrix[route_i[i_idx], route_i[i_idx+1]] + distance_matrix[route_j[j_idx], route_j[j_idx+1]]
                            new_cost = distance_matrix[route_i[i_idx], route_j[j_idx+1]] + distance_matrix[route_j[j_idx], route_i[i_idx+1]]
                            if new_cost >= old_cost - 1e-12:
                                continue
                            new_route_i = route_i[:i_idx+1] + route_j[j_idx+1:]
                            new_route_j = route_j[:j_idx+1] + route_i[i_idx+1:]
                            new_d_i = route_distance(new_route_i)
                            new_d_j = route_distance(new_route_j)
                            other_dists = [dists[idx] for idx in range(len(routes)) if idx not in (i_route, j_route)]
                            combined = other_dists + [new_d_i, new_d_j]
                            candidate_max = max(combined)
                            candidate_total = total_dist - dists[i_route] - dists[j_route] + new_d_i + new_d_j
                            if (candidate_max < best_new_max - 1e-12) or (abs(candidate_max - best_new_max) < 1e-12 and candidate_total < best_new_total - 1e-12):
                                best_new_max = candidate_max
                                best_new_total = candidate_total
                                best_move = ('2opt_star', i_route, i_idx, j_route, j_idx, new_route_i, new_route_j, new_d_i, new_d_j)
                if best_move is not None:
                    if best_move[0] == 'relocate':
                        _, i_route, pos, j_route, insert_pos, new_dist_i, new_dist_j = best_move
                        customer = routes[i_route].pop(pos)
                        dists[i_route] = new_dist_i
                        routes[j_route].insert(insert_pos, customer)
                        dists[j_route] = new_dist_j
                    elif best_move[0] == 'swap':
                        _, i_route, pos_i, j_route, pos_j, new_dist_i, new_dist_j = best_move
                        cust_i = routes[i_route][pos_i]
                        cust_j = routes[j_route][pos_j]
                        routes[i_route][pos_i] = cust_j
                        routes[j_route][pos_j] = cust_i
                        dists[i_route] = new_dist_i
                        dists[j_route] = new_dist_j
                    else:  # 2opt_star
                        _, i_route, i_idx, j_route, j_idx, new_route_i, new_route_j, new_d_i, new_d_j = best_move
                        routes[i_route] = new_route_i
                        routes[j_route] = new_route_j
                        dists[i_route] = new_d_i
                        dists[j_route] = new_d_j
                    total_dist = best_new_total
                    max_dist = best_new_max
                    report_best_vrp(routes)
                    improved = True
                    break
            if not improved:
                break
        return routes, dists, total_dist, max_dist

    def perturbation(routes, dists, remove_ratio):
        total_dist = sum(dists)
        max_dist = max(dists)
        n_removed = max(1, min(int(remove_ratio * (n-1)), n-1))
        max_d = max_dist
        weights = []
        for d in dists:
            if max_d > 0:
                weights.append(d / max_d)
            else:
                weights.append(0.0)
        total_weight = sum(weights)
        if total_weight == 0:
            probs = [1.0/len(weights)] * len(weights)
        else:
            probs = [w / total_weight for w in weights]
        customers = list(range(1, n))
        removed_set = set()
        while len(removed_set) < n_removed:
            route_idx = random.choices(range(len(routes)), weights=probs, k=1)[0]
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
        # Regret-3 insertion: for each customer, compute best, second best, third best max after insertion
        while removed_list:
            best_cust = None
            best_regret = -1
            best_route = None
            best_pos = None
            best_max_after = None
            for cust in removed_list:
                max_candidates = []
                for idx, route in enumerate(routes):
                    if len(route) == 1:  # empty route [0,0], length 2? Actually [0,0] has len 2
                        # empty route: insert at position 1
                        # but we have [0,0] so between 0 and 0
                        insert_cost = distance_matrix[0, cust] + distance_matrix[cust, 0]
                        new_dist = insert_cost
                        other_max = max([dists[j] for j in range(len(routes)) if j != idx])
                        candidate_max = max(other_max, new_dist)
                        max_candidates.append((candidate_max, idx, 1, new_dist))
                    else:
                        for k in range(1, len(route)):
                            pred = route[k-1]
                            succ = route[k]
                            insert_cost = distance_matrix[pred, cust] + distance_matrix[cust, succ] - distance_matrix[pred, succ]
                            new_dist = dists[idx] + insert_cost
                            other_max = max([dists[j] for j in range(len(routes)) if j != idx])
                            candidate_max = max(other_max, new_dist)
                            max_candidates.append((candidate_max, idx, k, new_dist))
                if not max_candidates:
                    continue
                max_candidates.sort(key=lambda x: x[0])
                if len(max_candidates) >= 3:
                    regret = max_candidates[2][0] - max_candidates[0][0]
                elif len(max_candidates) == 2:
                    regret = max_candidates[1][0] - max_candidates[0][0]
                else:
                    regret = 0
                if regret > best_regret - 1e-12:
                    best_regret = regret
                    best_cust = cust
                    best_max_after = max_candidates[0][0]
                    best_route = max_candidates[0][1]
                    best_pos = max_candidates[0][2]
                    best_new_dist = max_candidates[0][3]
            if best_cust is None:
                break
            # Insert best_cust at best_route best_pos
            routes[best_route].insert(best_pos, best_cust)
            dists[best_route] = best_new_dist
            total_dist = sum(dists)
            max_dist = max(dists)
            report_best_vrp(routes)
            removed_list.remove(best_cust)
        return routes, dists, total_dist, max_dist

    best_routes = None
    best_max = math.inf
    best_total = math.inf
    # Try multiple constructions with random order
    for _ in range(5):
        # Initialize routes with depot only
        routes = [[0, 0] for _ in range(truck_count)]
        dists = [0.0] * truck_count
        customers = list(range(1, n))
        random.shuffle(customers)
        # Insert each customer into the route that minimizes the resulting max distance
        for cust in customers:
            best_max_after = math.inf
            best_route = None
            best_pos = None
            best_new_dist = None
            for idx, route in enumerate(routes):
                if len(route) == 2 and route[0] == 0 and route[1] == 0:
                    # empty route
                    insert_cost = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    new_dist = insert_cost
                    other_max = max([dists[j] for j in range(len(routes)) if j != idx])
                    candidate_max = max(other_max, new_dist)
                    if candidate_max < best_max_after - 1e-12:
                        best_max_after = candidate_max
                        best_route = idx
                        best_pos = 1
                        best_new_dist = new_dist
                else:
                    for k in range(1, len(route)):
                        pred = route[k-1]
                        succ = route[k]
                        insert_cost = distance_matrix[pred, cust] + distance_matrix[cust, succ] - distance_matrix[pred, succ]
                        new_dist = dists[idx] + insert_cost
                        other_max = max([dists[j] for j in range(len(routes)) if j != idx])
                        candidate_max = max(other_max, new_dist)
                        if candidate_max < best_max_after - 1e-12:
                            best_max_after = candidate_max
                            best_route = idx
                            best_pos = k
                            best_new_dist = new_dist
            if best_route is not None:
                routes[best_route].insert(best_pos, cust)
                dists[best_route] = best_new_dist
        total_dist = sum(dists)
        max_dist = max(dists)
        report_best_vrp(routes)
        # Improve
        routes, dists, total_dist, max_dist = improvement(routes, dists)
        # Perturbation cycles
        for cycle in range(3):
            remove_ratio = 0.3 - cycle * 0.1
            if n > 2:
                routes, dists, total_dist, max_dist = perturbation(routes, dists, remove_ratio)
                routes, dists, total_dist, max_dist = improvement(routes, dists)
        if max_dist < best_max - 1e-12 or (abs(max_dist - best_max) < 1e-12 and total_dist < best_total - 1e-12):
            best_max = max_dist
            best_total = total_dist
            best_routes = [route[:] for route in routes]
    # Final improvement on best
    if best_routes is not None:
        best_routes, best_dists, _, _ = improvement(best_routes, compute_dists(best_routes))
        report_best_vrp(best_routes)
    return best_routes