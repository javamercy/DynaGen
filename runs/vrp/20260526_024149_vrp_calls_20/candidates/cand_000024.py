import numpy as np
import random

random.seed(0)

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def two_opt(route, dm):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                if j-i == 1:
                    continue
                old = dm[route[i-1], route[i]] + dm[route[j], route[j+1]]
                new = dm[route[i-1], route[j]] + dm[route[i], route[j+1]]
                if new < old - 1e-9:
                    route[i:j+1] = route[i:j+1][::-1]
                    improved = True
        # only one pass for efficiency? but we do full 2-opt
    return route

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= len(customers):
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    
    # Initialize each customer as a separate route
    routes = [[0, c, 0] for c in customers]
    
    # Clarke-Wright savings merging
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
    
    # Initial distances
    dists = [route_distance(r, distance_matrix) for r in routes]
    best_routes = [list(r) for r in routes]
    best_max = max(dists)
    report_best_vrp(best_routes)
    
    # Improvement loop
    max_iter = n * truck_count * 10
    stagnation = 0
    while stagnation < max_iter:
        dists = [route_distance(r, distance_matrix) for r in routes]
        max_dist = max(dists)
        if max_dist < best_max - 1e-9:
            best_max = max_dist
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
            stagnation = 0
        else:
            stagnation += 1
        # Find index of longest route (deterministic: first if tie)
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
        # Swap moves
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
        # If no improvement, apply intra-route 2-opt on longest route
        if not improved:
            # Apply 2-opt on longest route
            route_before = list(routes[max_idx])
            routes[max_idx] = two_opt(routes[max_idx], distance_matrix)
            if routes[max_idx] != route_before:
                improved = True
        # If still no improvement, perturb
        if not improved:
            # Perturb: move 2 random customers from longest to random positions in other routes
            if len(routes[max_idx]) > 4:
                # select two random distinct positions in longest route (exclude depot)
                longest_len = len(routes[max_idx])
                if longest_len >= 5:  # at least 2 customers with depots
                    positions = random.sample(range(1, longest_len-1), min(2, longest_len-2))
                    for pos in sorted(positions, reverse=True):
                        cust = routes[max_idx].pop(pos)
                        # choose random other route and random insert position
                        other_idx = random.choice([i for i in range(truck_count) if i != max_idx and len(routes[i]) >= 2])
                        insert_pos = random.randint(1, len(routes[other_idx])-1)
                        routes[other_idx].insert(insert_pos, cust)
                    improved = True
        if not improved:
            break
    # Final report and return
    dists = [route_distance(r, distance_matrix) for r in routes]
    if max(dists) < best_max - 1e-9:
        best_routes = [list(r) for r in routes]
        report_best_vrp(best_routes)
    else:
        report_best_vrp(best_routes)
    return best_routes