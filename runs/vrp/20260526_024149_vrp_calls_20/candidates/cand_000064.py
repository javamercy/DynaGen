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
    
    # Clarke-Wright savings initialization with tie-breaking
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
                # tie-breaking: prefer smaller first customer index in the pair
                if s1 > best_saving or (abs(s1 - best_saving) < 1e-9 and first_i < best_pair[0] if best_pair else True):
                    best_saving = s1
                    best_pair = (i, j)
                    best_order = 0
                if s2 > best_saving or (abs(s2 - best_saving) < 1e-9 and first_j < best_pair[0] if best_pair else True):
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
                best_cust = None
                best_pos = None
                best_other_idx = None
                best_insert_pos = None
                best_new_max = None
                best_new_other = None
                best_improvement = 0
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
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_cust = cust
                                best_pos = pos
                                best_other_idx = other_idx
                                best_insert_pos = insert_pos
                                best_new_max = new_max_route
                                best_new_other = new_other_route
                if best_improvement > 1e-9:
                    routes[max_idx] = best_new_max
                    routes[best_other_idx] = best_new_other
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
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_move = (max_idx, other_idx, pos_max, pos_other, new_max_route, new_other_route)
                if best_improvement > 1e-9:
                    _, other_idx, _, _, new_max_route, new_other_route = best_move
                    routes[max_idx] = new_max_route
                    routes[other_idx] = new_other_route
                    improved = True
            if improved:
                continue
            # Inter-route 2-opt*
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
                            new_r_max = [0] + r_max[1:i+1] + other_route[j+1:-1] + [0]
                            new_other = [0] + other_route[1:j+1] + r_max[i+1:-1] + [0]
                            new_max_dist = route_distance(new_r_max, distance_matrix)
                            new_other_dist = route_distance(new_other, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            improvement = max_dist - new_max
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_move = (max_idx, other_idx, new_r_max, new_other)
                if best_improvement > 1e-9:
                    _, other_idx, new_r_max, new_other = best_move
                    routes[max_idx] = new_r_max
                    routes[other_idx] = new_other
                    improved = True
            if improved:
                continue
            # Or-opt
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
                                if improvement > best_improvement:
                                    best_improvement = improvement
                                    best_move = (max_idx, other_idx, new_max_route, new_other_route)
                if best_improvement > 1e-9:
                    _, other_idx, new_max_route, new_other_route = best_move
                    routes[max_idx] = new_max_route
                    routes[other_idx] = new_other_route
                    improved = True
        return routes
    
    def simple_rebalance(routes):
        # Iteratively move customers from longest route to shortest route using best-improving insertions
        dists = [route_distance(r, distance_matrix) for r in routes]
        max_iters = n  # bounded
        for _ in range(max_iters):
            max_dist = max(dists)
            min_dist = min(dists)
            if max_dist - min_dist < 1e-9:
                break
            longest_idx = dists.index(max_dist)
            shortest_idx = dists.index(min_dist)
            if len(routes[longest_idx]) <= 2:
                break
            # Try moving each customer from longest route to shortest route, choose best improvement in max distance
            best_improvement = 0
            best_move = None
            r_long = routes[longest_idx]
            r_short = routes[shortest_idx]
            for pos in range(1, len(r_long)-1):
                cust = r_long[pos]
                new_long = r_long[:pos] + r_long[pos+1:]
                new_long_dist = route_distance(new_long, distance_matrix)
                for insert_pos in range(1, len(r_short)):
                    new_short = r_short[:insert_pos] + [cust] + r_short[insert_pos:]
                    new_short_dist = route_distance(new_short, distance_matrix)
                    new_max = max(new_long_dist, new_short_dist)
                    improvement = max_dist - new_max
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (longest_idx, shortest_idx, pos, insert_pos, new_long, new_short)
            if best_improvement > 1e-9:
                _, _, _, _, new_long, new_short = best_move
                routes[longest_idx] = new_long
                routes[shortest_idx] = new_short
                dists = [route_distance(r, distance_matrix) for r in routes]
            else:
                break
        return routes
    
    # Main loop with restarts and simulated annealing acceptance
    max_restarts = n
    plateau_count = 0
    ejection_frac = 0.1
    max_ejection_frac = 0.3
    temperature = 5.0
    cooling_rate = 0.92
    for restart in range(max_restarts):
        routes = local_search(routes, n * truck_count)
        dists = [route_distance(r, distance_matrix) for r in routes]
        current_max = max(dists)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
            plateau_count = 0
            ejection_frac = 0.1
            temperature = 5.0
        else:
            delta = current_max - best_max
            accept_worse = False
            if delta > 0 and np.random.random() < np.exp(-delta / temperature):
                accept_worse = True
            if accept_worse:
                best_max = current_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
                plateau_count += 1
            else:
                plateau_count += 1
            temperature *= cooling_rate
            if plateau_count >= 5:
                ejection_frac = min(ejection_frac + 0.05, max_ejection_frac)
                plateau_count = 0
            # Perturbation: distance-aware ejection chain
            max_idx = dists.index(current_max)
            r_max = routes[max_idx]
            if len(r_max) <= 3:
                break
            contributions = []
            for k in range(1, len(r_max)-1):
                prev = r_max[k-1]
                curr = r_max[k]
                next_cust = r_max[k+1]
                contrib = distance_matrix[prev][curr] + distance_matrix[curr][next_cust] - distance_matrix[prev][next_cust]
                contributions.append((contrib, k, curr))
            contributions.sort(reverse=True)
            num_eject = max(1, int((len(r_max)-2) * ejection_frac))
            ejected = [c[2] for c in contributions[:num_eject]]
            new_route = r_max.copy()
            for cust in ejected:
                new_route.remove(cust)
            # Greedy insertion of ejected customers into other routes
            for cust in ejected:
                best_increase = np.inf
                best_route_idx = -1
                best_pos = -1
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for pos in range(1, len(other_route)):
                        new_other_route = other_route[:pos] + [cust] + other_route[pos:]
                        new_dist = route_distance(new_other_route, distance_matrix)
                        old_dist = route_distance(other_route, distance_matrix)
                        increase = new_dist - old_dist
                        if increase < best_increase:
                            best_increase = increase
                            best_route_idx = other_idx
                            best_pos = pos
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
            routes[max_idx] = new_route
            # Simple rebalancing
            routes = simple_rebalance(routes)
    report_best_vrp(best_routes)
    return best_routes