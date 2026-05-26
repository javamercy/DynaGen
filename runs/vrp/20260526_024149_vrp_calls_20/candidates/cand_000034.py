import numpy as np
import random

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def two_opt(route, dm):
    improved = True
    best_route = route[:]
    best_dist = route_distance(best_route, dm)
    while improved:
        improved = False
        for i in range(1, len(best_route)-2):
            for j in range(i+1, len(best_route)-1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_dist = route_distance(new_route, dm)
                if new_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    return best_route

def perturb(routes, truck_count, n, dm, max_dist):
    dists = [route_distance(r, dm) for r in routes]
    longest_idx = max(range(truck_count), key=lambda i: dists[i])
    longest_route = routes[longest_idx]
    num_eject = min(3, len(longest_route)-2)
    if num_eject <= 0:
        return routes
    savings = []
    for k in range(1, len(longest_route)-1):
        prev = longest_route[k-1]
        cur = longest_route[k]
        nxt = longest_route[k+1]
        saving = dm[prev][cur] + dm[cur][nxt] - dm[prev][nxt]
        savings.append((saving, k, cur))
    savings.sort(reverse=True, key=lambda x: x[0])
    selected = [savings[i][2] for i in range(num_eject)]
    selected_positions = sorted([savings[i][1] for i in range(num_eject)], reverse=True)
    new_route = longest_route[:]
    for pos in selected_positions:
        new_route = new_route[:pos] + new_route[pos+1:]
    routes_new = [r[:] for r in routes]
    routes_new[longest_idx] = new_route
    for cust in selected:
        best_new_max = float('inf')
        best_insert = None
        current_dists = [route_distance(r, dm) for r in routes_new]
        for r_idx in range(truck_count):
            route = routes_new[r_idx]
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance(new_route, dm)
                if r_idx == longest_idx:
                    new_max = max(max(current_dists[r_idx2] for r_idx2 in range(truck_count) if r_idx2 != r_idx), new_dist)
                else:
                    other_dists = [current_dists[i] if i != r_idx else new_dist for i in range(truck_count)]
                    new_max = max(other_dists)
                if new_max < best_new_max - 1e-9:
                    best_new_max = new_max
                    best_insert = (r_idx, pos)
        if best_insert is not None:
            r_idx, pos = best_insert
            route = routes_new[r_idx]
            routes_new[r_idx] = route[:pos] + [cust] + route[pos:]
    return routes_new

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= len(customers):
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
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
    
    random.seed(0)
    max_restarts = min(3, n * truck_count // 10 + 1)
    current_routes = [list(r) for r in routes]
    
    for restart in range(max_restarts):
        max_iter = n * truck_count
        for iteration in range(max_iter):
            dists = [route_distance(r, distance_matrix) for r in current_routes]
            max_dist = max(dists)
            if max_dist < best_max - 1e-9:
                best_max = max_dist
                best_routes = [list(r) for r in current_routes]
                report_best_vrp(best_routes)
            
            max_idx = dists.index(max_dist)
            improved = False
            
            # Relocate moves from longest route
            if len(current_routes[max_idx]) > 2:
                for pos in range(1, len(current_routes[max_idx])-1):
                    cust = current_routes[max_idx][pos]
                    new_max_route = current_routes[max_idx][:pos] + current_routes[max_idx][pos+1:]
                    new_max_dist = route_distance(new_max_route, distance_matrix)
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = current_routes[other_idx]
                        for insert_pos in range(1, len(other_route)):
                            new_other_route = other_route[:insert_pos] + [cust] + other_route[insert_pos:]
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            if new_max < max_dist - 1e-9:
                                current_routes[max_idx] = new_max_route
                                current_routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # If no relocate improvement, try swap moves
            if not improved and len(current_routes[max_idx]) > 2:
                for other_idx in range(truck_count):
                    if other_idx == max_idx or len(current_routes[other_idx]) <= 2:
                        continue
                    for pos_max in range(1, len(current_routes[max_idx])-1):
                        cust_a = current_routes[max_idx][pos_max]
                        for pos_other in range(1, len(current_routes[other_idx])-1):
                            cust_b = current_routes[other_idx][pos_other]
                            new_max_route = current_routes[max_idx].copy()
                            new_max_route[pos_max] = cust_b
                            new_max_dist = route_distance(new_max_route, distance_matrix)
                            new_other_route = current_routes[other_idx].copy()
                            new_other_route[pos_other] = cust_a
                            new_other_dist = route_distance(new_other_route, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max = max(new_dists)
                            if new_max < max_dist - 1e-9:
                                current_routes[max_idx] = new_max_route
                                current_routes[other_idx] = new_other_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Cross-route 2-opt* exchange (added)
            if not improved and len(current_routes[max_idx]) > 2:
                for other_idx in range(truck_count):
                    if other_idx == max_idx or len(current_routes[other_idx]) <= 2:
                        continue
                    max_route = current_routes[max_idx]
                    other_route = current_routes[other_idx]
                    for i in range(1, len(max_route)-1):
                        for j in range(1, len(other_route)-1):
                            # swap tails: max_route[0:i] + other_route[j:] and other_route[0:j] + max_route[i:]
                            new_max = max_route[:i] + other_route[j:]
                            new_other = other_route[:j] + max_route[i:]
                            # Ensure routes end at depot
                            if new_max[-1] != 0 or new_other[-1] != 0:
                                continue
                            new_max_dist = route_distance(new_max, distance_matrix)
                            new_other_dist = route_distance(new_other, distance_matrix)
                            new_dists = dists.copy()
                            new_dists[max_idx] = new_max_dist
                            new_dists[other_idx] = new_other_dist
                            new_max_val = max(new_dists)
                            if new_max_val < max_dist - 1e-9:
                                current_routes[max_idx] = new_max
                                current_routes[other_idx] = new_other
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Intra-route 2-opt on longest route if no improvement
            if not improved and len(current_routes[max_idx]) > 2:
                old_route = current_routes[max_idx]
                new_route = two_opt(old_route, distance_matrix)
                if route_distance(new_route, distance_matrix) < route_distance(old_route, distance_matrix) - 1e-9:
                    current_routes[max_idx] = new_route
                    improved = True
            
            if not improved:
                break
        
        final_dists = [route_distance(r, distance_matrix) for r in current_routes]
        current_max = max(final_dists)
        if current_max < best_max - 1e-9:
            best_max = current_max
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
        
        if restart < max_restarts - 1:
            current_routes = perturb(current_routes, truck_count, n, distance_matrix, best_max)
    
    report_best_vrp(best_routes)
    return best_routes