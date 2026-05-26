import numpy as np
import random

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    # trivial case: each customer on its own route
    if truck_count >= len(customers):
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # Initialize each customer as a separate route
    routes = [[0, c, 0] for c in customers]
    
    # Merge using Clarke-Wright savings until truck_count routes remain
    # Use random tie-breaking
    while len(routes) > truck_count:
        savings = []
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
                savings.append((s1, i, j, 0))
                savings.append((s2, i, j, 1))
        if not savings:
            break
        # find max saving (random among ties)
        max_saving = max(s[0] for s in savings)
        candidates = [s for s in savings if s[0] == max_saving]
        selected = random.choice(candidates)
        _, i, j, order = selected
        if order == 0:
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
    
    # Intra-route 2-opt on each route
    for idx, route in enumerate(routes):
        if len(route) <= 3:
            continue
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route, distance_matrix) < route_distance(route, distance_matrix):
                        route = new_route
                        routes[idx] = route
                        improved = True
    
    # Compute initial distances and report
    dists = [route_distance(r, distance_matrix) for r in routes]
    best_routes = [list(r) for r in routes]
    best_max = max(dists)
    report_best_vrp(best_routes)
    
    # Improvement phase: relocate and swap moves focusing on max route
    import itertools
    max_iter = n * truck_count * 2
    no_improve_count = 0
    perturbation_rounds = 0
    max_perturb = 3
    for _ in range(max_iter):
        dists = [route_distance(r, distance_matrix) for r in routes]
        max_dist = max(dists)
        if max_dist < best_max - 1e-9:
            best_max = max_dist
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        # find index of longest route (first if tie)
        max_idx = dists.index(max_dist)
        improved = False
        # Relocate moves from longest route
        if len(routes[max_idx]) > 2:
            for pos in range(1, len(routes[max_idx])-1):
                cust = routes[max_idx][pos]
                new_max_route = routes[max_idx][:pos] + routes[max_idx][pos+1:]
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
        # If no relocate improvement, try swap moves between longest and another route
        if not improved and len(routes[max_idx]) > 2:
            for other_idx in range(truck_count):
                if other_idx == max_idx or len(routes[other_idx]) <= 2:
                    continue
                for pos_max in range(1, len(routes[max_idx])-1):
                    cust_a = routes[max_idx][pos_max]
                    for pos_other in range(1, len(routes[other_idx])-1):
                        cust_b = routes[other_idx][pos_other]
                        new_max_route = routes[max_idx].copy()
                        new_max_route[pos_max] = cust_b
                        new_max_dist = route_distance(new_max_route, distance_matrix)
                        new_other_route = routes[other_idx].copy()
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
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 5 and perturbation_rounds < max_perturb:
                # Perturbation: randomly relocate up to 3 customers from longest route to other routes
                long_route = routes[max_idx]
                if len(long_route) > 2:
                    positions = list(range(1, len(long_route)-1))
                    random.shuffle(positions)
                    num_reloc = min(3, len(positions))
                    for p in positions[:num_reloc]:
                        cust = long_route[p]
                        new_long = long_route[:p] + long_route[p+1:]
                        # find best insertion in other routes (minimize max distance)
                        best_other_idx = -1
                        best_insert_pos = -1
                        best_new_max = float('inf')
                        for other_idx in range(truck_count):
                            if other_idx == max_idx:
                                continue
                            other_route = routes[other_idx]
                            for insert_pos in range(1, len(other_route)):
                                new_other = other_route[:insert_pos] + [cust] + other_route[insert_pos:]
                                new_dists = dists.copy()
                                new_dists[max_idx] = route_distance(new_long, distance_matrix)
                                new_dists[other_idx] = route_distance(new_other, distance_matrix)
                                new_max = max(new_dists)
                                if new_max < best_new_max:
                                    best_new_max = new_max
                                    best_other_idx = other_idx
                                    best_insert_pos = insert_pos
                        if best_other_idx != -1:
                            # perform move
                            routes[max_idx] = new_long
                            other_route = routes[best_other_idx]
                            new_other = other_route[:best_insert_pos] + [cust] + other_route[best_insert_pos:]
                            routes[best_other_idx] = new_other
                            dists = [route_distance(r, distance_matrix) for r in routes]
                            long_route = routes[max_idx]
                perturbation_rounds += 1
                no_improve_count = 0
    # Final report and return
    report_best_vrp(best_routes)
    return best_routes