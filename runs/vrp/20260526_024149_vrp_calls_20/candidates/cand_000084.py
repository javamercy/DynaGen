import numpy as np
import random
import math

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
    
    best_routes = None
    best_max = float('inf')
    
    # Multi-start with shuffled customer order
    for start_phase in range(5):
        shuffled_cust = customers[:]
        random.shuffle(shuffled_cust)
        # Clarke-Wright savings initialization
        routes = [[0, c, 0] for c in shuffled_cust]
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
        
        # Compute initial max distance
        dists = [route_distance(r, distance_matrix) for r in routes]
        current_max = max(dists)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        
        # Local search function with route-balancing penalty
        def local_search(routes, max_iter, temperature=0.0):
            nonlocal best_max, best_routes
            improved = True
            for _ in range(max_iter):
                if not improved:
                    break
                improved = False
                dists = [route_distance(r, distance_matrix) for r in routes]
                max_dist = max(dists)
                max_idx = dists.index(max_dist)
                n_routes = len(routes)
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
                    # Update max_dist after change
                    dists = [route_distance(r, distance_matrix) for r in routes]
                    max_dist = max(dists)
                    if max_dist < best_max - 1e-12:
                        best_max = max_dist
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
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
                                new_max_candidate = max(new_max_dist, new_other_dist, max(dists[:max_idx] + dists[max_idx+1:other_idx] + dists[other_idx+1:]))
                                # Accept if strict improvement or with SA probability
                                if new_max_candidate < max_dist - 1e-9:
                                    routes[max_idx] = new_max_route
                                    routes[other_idx] = new_other_route
                                    improved = True
                                    break
                                elif temperature > 0 and new_max_candidate < max_dist:
                                    delta = max_dist - new_max_candidate
                                    if random.random() < math.exp(delta / temperature):
                                        routes[max_idx] = new_max_route
                                        routes[other_idx] = new_other_route
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    dists = [route_distance(r, distance_matrix) for r in routes]
                    max_dist = max(dists)
                    if max_dist < best_max - 1e-12:
                        best_max = max_dist
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
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
                                new_max_candidate = max(new_max_dist, new_other_dist, max(dists[:max_idx] + dists[max_idx+1:other_idx] + dists[other_idx+1:]))
                                if new_max_candidate < max_dist - 1e-9:
                                    routes[max_idx] = new_max_route
                                    routes[other_idx] = new_other_route
                                    improved = True
                                    break
                                elif temperature > 0 and new_max_candidate < max_dist:
                                    delta = max_dist - new_max_candidate
                                    if random.random() < math.exp(delta / temperature):
                                        routes[max_idx] = new_max_route
                                        routes[other_idx] = new_other_route
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    dists = [route_distance(r, distance_matrix) for r in routes]
                    max_dist = max(dists)
                    if max_dist < best_max - 1e-12:
                        best_max = max_dist
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                    continue
                # Inter-route 2-opt*
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
                                new_max_candidate = max(new_max_dist, new_other_dist, max(dists[:max_idx] + dists[max_idx+1:other_idx] + dists[other_idx+1:]))
                                if new_max_candidate < max_dist - 1e-9:
                                    routes[max_idx] = new_r_max
                                    routes[other_idx] = new_other
                                    improved = True
                                    break
                                elif temperature > 0 and new_max_candidate < max_dist:
                                    delta = max_dist - new_max_candidate
                                    if random.random() < math.exp(delta / temperature):
                                        routes[max_idx] = new_r_max
                                        routes[other_idx] = new_other
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    dists = [route_distance(r, distance_matrix) for r in routes]
                    max_dist = max(dists)
                    if max_dist < best_max - 1e-12:
                        best_max = max_dist
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                    continue
                # Or-opt from longest route
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
                                    new_max_candidate = max(new_max_dist, new_other_dist, max(dists[:max_idx] + dists[max_idx+1:other_idx] + dists[other_idx+1:]))
                                    if new_max_candidate < max_dist - 1e-9:
                                        routes[max_idx] = new_max_route
                                        routes[other_idx] = new_other_route
                                        improved = True
                                        break
                                    elif temperature > 0 and new_max_candidate < max_dist:
                                        delta = max_dist - new_max_candidate
                                        if random.random() < math.exp(delta / temperature):
                                            routes[max_idx] = new_max_route
                                            routes[other_idx] = new_other_route
                                            improved = True
                                            break
                                if improved:
                                    break
                            if improved:
                                break
                if improved:
                    dists = [route_distance(r, distance_matrix) for r in routes]
                    max_dist = max(dists)
                    if max_dist < best_max - 1e-12:
                        best_max = max_dist
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
            return routes
        
        # Main loop with restart, perturbation, SA
        max_restarts = n
        plateau_count = 0
        ejection_frac = 0.1
        max_ejection_frac = 0.3
        temperature = 1.0
        cooling_rate = 0.99
        for restart in range(max_restarts):
            routes = local_search(routes, n * truck_count, temperature)
            dists = [route_distance(r, distance_matrix) for r in routes]
            current_max = max(dists)
            if current_max < best_max - 1e-12:
                best_max = current_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
                plateau_count = 0
                ejection_frac = 0.1
                temperature = 1.0
            else:
                plateau_count += 1
                if plateau_count >= 5:
                    ejection_frac = min(ejection_frac + 0.05, max_ejection_frac)
                    plateau_count = 0
                temperature *= cooling_rate
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
                # Greedy rebalancing: insert ejected customers into routes with smallest current distance (or smallest max?)
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
            # Periodic restart using shuffled elite subset (keep best routes, shuffle others)
            if restart % 10 == 0 and restart > 0:
                # Keep best 2 routes (or all if less than 2)
                elite_count = min(2, truck_count)
                dists_with_idx = [(route_distance(r, distance_matrix), i) for i, r in enumerate(routes)]
                dists_with_idx.sort()
                elite_indices = [d[1] for d in dists_with_idx[:elite_count]]
                elite_routes = [list(routes[i]) for i in elite_indices]
                other_indices = [i for i in range(truck_count) if i not in elite_indices]
                # Shuffle other customers (non-depot nodes not in elite routes)
                elite_cust = set()
                for r in elite_routes:
                    for c in r[1:-1]:
                        elite_cust.add(c)
                remaining_cust = [c for c in customers if c not in elite_cust]
                random.shuffle(remaining_cust)
                # Rebuild other routes using remaining customers
                # Simple assignment: first come first serve but ensure each route has at least one customer? We'll just distribute
                k = 0
                new_other_routes = []
                for idx in other_indices:
                    if k < len(remaining_cust):
                        route = [0, remaining_cust[k], 0]
                        k += 1
                        new_other_routes.append(route)
                    else:
                        new_other_routes.append([0, 0])
                # Assign remaining customers to these routes (simple insertion)
                while k < len(remaining_cust):
                    cust = remaining_cust[k]
                    # Find route with smallest distance increase
                    best_inc = np.inf
                    best_route_idx = -1
                    best_pos = -1
                    for idx_in_list, route in enumerate(new_other_routes):
                        if len(route) <= 2:
                            # Empty route, just insert at position 1
                            if 0 < best_inc:
                                best_inc = 0
                                best_route_idx = idx_in_list
                                best_pos = 1
                            continue
                        for pos in range(1, len(route)):
                            new_route = route[:pos] + [cust] + route[pos:]
                            new_dist = route_distance(new_route, distance_matrix)
                            old_dist = route_distance(route, distance_matrix)
                            inc = new_dist - old_dist
                            if inc < best_inc:
                                best_inc = inc
                                best_route_idx = idx_in_list
                                best_pos = pos
                    new_other_routes[best_route_idx] = new_other_routes[best_route_idx][:best_pos] + [cust] + new_other_routes[best_route_idx][best_pos:]
                    k += 1
                # Combine elite and other routes
                new_routes = [None] * truck_count
                for i, idx in enumerate(elite_indices):
                    new_routes[idx] = elite_routes[i]
                for i, idx in enumerate(other_indices):
                    new_routes[idx] = new_other_routes[i]
                routes = new_routes
                # Evaluate new max
                dists = [route_distance(r, distance_matrix) for r in routes]
                current_max = max(dists)
                if current_max < best_max - 1e-12:
                    best_max = current_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
    
    report_best_vrp(best_routes)
    return best_routes