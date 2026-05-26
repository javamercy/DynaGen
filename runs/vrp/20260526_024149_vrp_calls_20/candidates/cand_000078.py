import numpy as np

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def rebalance(routes, dm, truck_count):
    improved = True
    for _ in range(truck_count * 2):
        if not improved:
            break
        improved = False
        dists = [route_distance(r, dm) for r in routes]
        max_dist = max(dists)
        max_idx = dists.index(max_dist)
        r_max = routes[max_idx]
        if len(r_max) <= 3:
            continue
        for pos in range(1, len(r_max)-1):
            cust = r_max[pos]
            new_max_route = r_max[:pos] + r_max[pos+1:]
            new_max_dist = route_distance(new_max_route, dm)
            best_other_idx = -1
            best_insert_pos = -1
            best_new_other_dist = None
            best_new_max = None
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for insert_pos in range(1, len(other_route)):
                    new_other_route = other_route[:insert_pos] + [cust] + other_route[insert_pos:]
                    new_other_dist = route_distance(new_other_route, dm)
                    new_dists = dists.copy()
                    new_dists[max_idx] = new_max_dist
                    new_dists[other_idx] = new_other_dist
                    new_max = max(new_dists)
                    if new_max < max_dist - 1e-9:
                        if best_other_idx == -1 or new_max < best_new_max:
                            best_other_idx = other_idx
                            best_insert_pos = insert_pos
                            best_new_other_dist = new_other_dist
                            best_new_max = new_max
            if best_other_idx != -1:
                routes[max_idx] = new_max_route
                routes[best_other_idx] = routes[best_other_idx][:best_insert_pos] + [cust] + routes[best_other_idx][best_insert_pos:]
                improved = True
                break
    return routes

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
                            if new_max < max_dist - 1e-9:
                                routes[max_idx] = new_max_route
                                routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                continue
            # Inter-route swap
            if len(routes[max_idx]) > 2:
                r_max = routes[max_idx]
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
                            if new_max < max_dist - 1e-9:
                                routes[max_idx] = new_max_route
                                routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                continue
            # Inter-route 2-opt* (cross-exchange)
            if len(routes[max_idx]) > 2:
                r_max = routes[max_idx]
                for other_idx in range(truck_count):
                    if other_idx == max_idx or len(routes[other_idx]) <= 2:
                        continue
                    other_route = routes[other_idx]
                    for i in range(1, len(r_max)-2):
                        for j in range(1, len(other_route)-2):
                            new_r_max = r_max[:i+1] + other_route[j+1:-1] + [0]
                            new_other = other_route[:j+1] + r_max[i+1:-1] + [0]
                            new_r_max[0] = 0
                            new_other[0] = 0
                            new_r_max = [0] + new_r_max[1:]
                            new_other = [0] + new_other[1:]
                            new_max_dist = route_distance(new_r_max, distance_matrix)
                            new_other_dist = route_distance(new_other, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            if new_max < max_dist - 1e-9:
                                routes[max_idx] = new_r_max
                                routes[other_idx] = new_other
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                continue
            # Or-opt: move a block of 1-3 consecutive customers from longest to another route
            if len(routes[max_idx]) > 3:
                r_max = routes[max_idx]
                for block_len in range(1, min(4, len(r_max)-2)):
                    if improved:
                        break
                    for start in range(1, len(r_max)-block_len):
                        if improved:
                            break
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
                                if new_max < max_dist - 1e-9:
                                    routes[max_idx] = new_max_route
                                    routes[other_idx] = new_other_route
                                    improved = True
                                    break
                            if improved:
                                break
        return routes
    
    # Main loop with simulated annealing
    max_iter = n * truck_count
    temperature = best_max * 0.1
    cooling_rate = 0.95
    plateau_count = 0
    ejection_frac = 0.1
    max_ejection_frac = 0.3
    for restart in range(max_iter):
        routes = local_search(routes, n * truck_count)
        dists = [route_distance(r, distance_matrix) for r in routes]
        current_max = max(dists)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
            plateau_count = 0
            ejection_frac = 0.1
        else:
            # Simulated annealing acceptance
            delta = current_max - best_max
            if delta > 0 and np.random.rand() < np.exp(-delta / temperature):
                # Accept worse solution
                plateau_count += 1
                if plateau_count >= 5:
                    ejection_frac = min(ejection_frac + 0.05, max_ejection_frac)
                    plateau_count = 0
            else:
                # Reject: revert to best solution
                routes = [list(r) for r in best_routes]
                plateau_count += 1
                if plateau_count >= 5:
                    ejection_frac = min(ejection_frac + 0.05, max_ejection_frac)
                    plateau_count = 0
                continue
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
        # Greedy rebalancing after perturbation
        routes = rebalance(routes, distance_matrix, truck_count)
        temperature *= cooling_rate
        if temperature < 1e-6:
            temperature = best_max * 0.1  # reheat
    report_best_vrp(best_routes)
    return best_routes