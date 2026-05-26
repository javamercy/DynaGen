import numpy as np

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= len(customers):
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # Clarke-Wright savings initialization
    routes = [[0, c, 0] for c in customers]
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
    
    def local_search(routes, max_iter):
        improved = True
        for _ in range(max_iter):
            if not improved:
                break
            improved = False
            dists = [route_distance(r, distance_matrix) for r in routes]
            max_dist = max(dists)
            max_idx = dists.index(max_dist)
            # Intra-route 2-opt on longest route
            if len(routes[max_idx]) > 3:
                r = routes[max_idx]
                best_imp = 0
                best_pair = None
                for i in range(1, len(r)-2):
                    for j in range(i+1, len(r)-1):
                        if j - i == 1:
                            continue
                        new_route = r[:i] + r[i:j+1][::-1] + r[j+1:]
                        new_dist = route_distance(new_route, distance_matrix)
                        old_dist = route_distance(r, distance_matrix)
                        if new_dist < old_dist - 1e-9:
                            improvement = old_dist - new_dist
                            if improvement > best_imp:
                                best_imp = improvement
                                best_pair = (i, j, new_route)
                if best_pair:
                    i, j, new_route = best_pair
                    routes[max_idx] = new_route
                    improved = True
            if improved:
                continue
            # Inter-route relocate from longest route
            if len(routes[max_idx]) > 2:
                r_max = routes[max_idx]
                best_improvement = 0
                best_move = None
                for pos in range(1, len(r_max)-1):
                    cust = r_max[pos]
                    new_max_route = r_max[:pos] + r_max[pos+1:]
                    new_max_dist = route_distance(new_max_route, distance_matrix)
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = routes[other_idx]
                        for insert_pos in range(1, len(other_route)):
                            new_other_route = other_route[:insert_pos] + [cust] + other_route[insert_pos:]
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            improvement = max_dist - new_max
                            if improvement > best_improvement + 1e-9:
                                best_improvement = improvement
                                best_move = (max_idx, other_idx, pos, insert_pos, cust)
                if best_move:
                    max_idx, other_idx, pos, insert_pos, cust = best_move
                    routes[max_idx] = routes[max_idx][:pos] + routes[max_idx][pos+1:]
                    routes[other_idx] = routes[other_idx][:insert_pos] + [cust] + routes[other_idx][insert_pos:]
                    improved = True
            if improved:
                continue
            # Inter-route swap
            if len(routes[max_idx]) > 2:
                r_max = routes[max_idx]
                best_improvement = 0
                best_move = None
                for other_idx in range(truck_count):
                    if other_idx == max_idx or len(routes[other_idx]) <= 2:
                        continue
                    other_route = routes[other_idx]
                    for pos_max in range(1, len(r_max)-1):
                        cust_a = r_max[pos_max]
                        for pos_other in range(1, len(other_route)-1):
                            cust_b = other_route[pos_other]
                            new_max_route = r_max.copy()
                            new_max_route[pos_max] = cust_b
                            new_max_dist = route_distance(new_max_route, distance_matrix)
                            new_other_route = other_route.copy()
                            new_other_route[pos_other] = cust_a
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            improvement = max_dist - new_max
                            if improvement > best_improvement + 1e-9:
                                best_improvement = improvement
                                best_move = (max_idx, other_idx, pos_max, pos_other, cust_a, cust_b)
                if best_move:
                    max_idx, other_idx, pos_max, pos_other, cust_a, cust_b = best_move
                    routes[max_idx][pos_max] = cust_b
                    routes[other_idx][pos_other] = cust_a
                    improved = True
            if improved:
                continue
            # Inter-route 2-opt* (cross-exchange)
            if len(routes[max_idx]) > 2:
                r_max = routes[max_idx]
                best_improvement = 0
                best_move = None
                for other_idx in range(truck_count):
                    if other_idx == max_idx or len(routes[other_idx]) <= 2:
                        continue
                    other_route = routes[other_idx]
                    for i in range(1, len(r_max)-2):
                        for j in range(1, len(other_route)-2):
                            # swap tails
                            new_r_max = r_max[:i+1] + other_route[j+1:-1] + [0]
                            new_other = other_route[:j+1] + r_max[i+1:-1] + [0]
                            new_r_max[0] = 0
                            new_other[0] = 0
                            new_r_max[0] = 0
                            new_other[0] = 0  # already set
                            new_max_dist = route_distance(new_r_max, distance_matrix)
                            new_other_dist = route_distance(new_other, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            improvement = max_dist - new_max
                            if improvement > best_improvement + 1e-9:
                                best_improvement = improvement
                                best_move = (max_idx, other_idx, i, j, new_r_max, new_other)
                if best_move:
                    max_idx, other_idx, i, j, new_r_max, new_other = best_move
                    routes[max_idx] = new_r_max
                    routes[other_idx] = new_other
                    improved = True
            if improved:
                continue
            # Or-opt: move a block of 1-3 consecutive customers from longest to another route
            if len(routes[max_idx]) > 3:
                r_max = routes[max_idx]
                best_improvement = 0
                best_move = None
                for block_len in range(1, min(4, len(r_max)-2)):
                    for start in range(1, len(r_max)-block_len):
                        block = r_max[start:start+block_len]
                        new_max_route = r_max[:start] + r_max[start+block_len:]
                        new_max_dist = route_distance(new_max_route, distance_matrix)
                        for other_idx in range(truck_count):
                            if other_idx == max_idx:
                                continue
                            other_route = routes[other_idx]
                            for insert_pos in range(1, len(other_route)):
                                new_other_route = other_route[:insert_pos] + block + other_route[insert_pos:]
                                new_other_dist = route_distance(new_other_route, distance_matrix)
                                new_dists = dists.copy()
                                new_dists[max_idx] = new_max_dist
                                new_dists[other_idx] = new_other_dist
                                new_max = max(new_dists)
                                improvement = max_dist - new_max
                                if improvement > best_improvement + 1e-9:
                                    best_improvement = improvement
                                    best_move = (max_idx, other_idx, start, insert_pos, block_len, block)
                if best_move:
                    max_idx, other_idx, start, insert_pos, block_len, block = best_move
                    routes[max_idx] = routes[max_idx][:start] + routes[max_idx][start+block_len:]
                    routes[other_idx] = routes[other_idx][:insert_pos] + block + routes[other_idx][insert_pos:]
                    improved = True
        return routes
    
    # Main loop with restarts and perturbation
    max_restarts = min(8, n)  # bounded
    threshold_start = 0.05
    for restart in range(max_restarts):
        routes = local_search(routes, n * truck_count)
        dists = [route_distance(r, distance_matrix) for r in routes]
        current_max = max(dists)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        else:
            # Perturbation: least-worsening moves accepting up to threshold
            threshold = threshold_start * (0.9 ** restart)  # decreasing threshold
            max_allowed = best_max * (1 + threshold)
            # Try relocate, swap, 2-opt* that worsen max but within threshold
            perturbed = False
            # We'll try multiple random perturbations? But we need deterministic. Use first-found worsening within threshold.
            # Use a loop to try to find a perturbation that stays within threshold.
            # For each type, we'll generate candidate moves.
            # Since we need deterministic, we'll iterate over possibilities in order.
            candidates = []
            # Relocate
            for i in range(truck_count):
                ri = routes[i]
                if len(ri) <= 2:
                    continue
                for pos in range(1, len(ri)-1):
                    cust = ri[pos]
                    new_ri = ri[:pos] + ri[pos+1:]
                    new_ri_dist = route_distance(new_ri, distance_matrix)
                    for j in range(truck_count):
                        if j == i:
                            continue
                        rj = routes[j]
                        for insert_pos in range(1, len(rj)):
                            new_rj = rj[:insert_pos] + [cust] + rj[insert_pos:]
                            new_rj_dist = route_distance(new_rj, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[i] = new_ri_dist
                            new_dists[j] = new_rj_dist
                            new_max = max(new_dists)
                            if new_max <= max_allowed and new_max > best_max:  # worsen but within threshold
                                candidates.append((new_max, 'relocate', i, j, pos, insert_pos, cust))
            # Swap
            for i in range(truck_count):
                ri = routes[i]
                if len(ri) <= 2:
                    continue
                for pos_i in range(1, len(ri)-1):
                    cust_a = ri[pos_i]
                    for j in range(i+1, truck_count):
                        rj = routes[j]
                        if len(rj) <= 2:
                            continue
                        for pos_j in range(1, len(rj)-1):
                            cust_b = rj[pos_j]
                            new_ri = ri.copy()
                            new_ri[pos_i] = cust_b
                            new_ri_dist = route_distance(new_ri, distance_matrix)
                            new_rj = rj.copy()
                            new_rj[pos_j] = cust_a
                            new_rj_dist = route_distance(new_rj, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[i] = new_ri_dist
                            new_dists[j] = new_rj_dist
                            new_max = max(new_dists)
                            if new_max <= max_allowed and new_max > best_max:
                                candidates.append((new_max, 'swap', i, j, pos_i, pos_j, cust_a, cust_b))
            # 2-opt*
            for i in range(truck_count):
                ri = routes[i]
                if len(ri) <= 2:
                    continue
                for j in range(i+1, truck_count):
                    rj = routes[j]
                    if len(rj) <= 2:
                        continue
                    for pos_i in range(1, len(ri)-2):
                        for pos_j in range(1, len(rj)-2):
                            new_ri = ri[:pos_i+1] + rj[pos_j+1:-1] + [0]
                            new_rj = rj[:pos_j+1] + ri[pos_i+1:-1] + [0]
                            new_ri[0] = 0
                            new_rj[0] = 0
                            new_ri_dist = route_distance(new_ri, distance_matrix)
                            new_rj_dist = route_distance(new_rj, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[i] = new_ri_dist
                            new_dists[j] = new_rj_dist
                            new_max = max(new_dists)
                            if new_max <= max_allowed and new_max > best_max:
                                candidates.append((new_max, '2opt_star', i, j, pos_i, pos_j, new_ri, new_rj))
            if candidates:
                # Choose the one with smallest new_max (least worsening)
                candidates.sort()
                chosen = candidates[0]
                if chosen[1] == 'relocate':
                    _, _, i, j, pos, insert_pos, cust = chosen
                    routes[i] = routes[i][:pos] + routes[i][pos+1:]
                    routes[j] = routes[j][:insert_pos] + [cust] + routes[j][insert_pos:]
                elif chosen[1] == 'swap':
                    _, _, i, j, pos_i, pos_j, cust_a, cust_b = chosen
                    routes[i][pos_i] = cust_b
                    routes[j][pos_j] = cust_a
                else:  # 2opt_star
                    _, _, i, j, pos_i, pos_j, new_ri, new_rj = chosen
                    routes[i] = new_ri
                    routes[j] = new_rj
                perturbed = True
            if not perturbed:
                # If no perturbation found, reinitialize from scratch with Clarke-Wright but shuffled? To keep deterministic, just break or do nothing.
                break
            # Greedy rebalancing: after perturbation, for each route, if its distance is close to best_max, try to move a customer to a shorter route
            dists = [route_distance(r, distance_matrix) for r in routes]
            current_max = max(dists)
            max_idx = dists.index(current_max)
            # Attempt to reduce max by moving a customer from max route to a short route
            if len(routes[max_idx]) > 2:
                r_max = routes[max_idx]
                best_reduction = 0
                best_move = None
                for pos in range(1, len(r_max)-1):
                    cust = r_max[pos]
                    new_max_route = r_max[:pos] + r_max[pos+1:]
                    new_max_dist = route_distance(new_max_route, distance_matrix)
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = routes[other_idx]
                        for insert_pos in range(1, len(other_route)):
                            new_other_route = other_route[:insert_pos] + [cust] + other_route[insert_pos:]
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            if new_max < current_max - 1e-9:
                                reduction = current_max - new_max
                                if reduction > best_reduction:
                                    best_reduction = reduction
                                    best_move = (max_idx, other_idx, pos, insert_pos, cust)
                if best_move:
                    max_idx, other_idx, pos, insert_pos, cust = best_move
                    routes[max_idx] = routes[max_idx][:pos] + routes[max_idx][pos+1:]
                    routes[other_idx] = routes[other_idx][:insert_pos] + [cust] + routes[other_idx][insert_pos:]
                    # Update dists and best_max if improved
                    dists = [route_distance(r, distance_matrix) for r in routes]
                    current_max = max(dists)
                    if current_max < best_max - 1e-9:
                        best_max = current_max
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
    report_best_vrp(best_routes)
    return best_routes