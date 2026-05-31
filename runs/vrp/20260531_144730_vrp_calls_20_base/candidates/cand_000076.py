import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)
    
    # ------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_route_dist(routes, dists):
        return max(dists)

    # ------------------------------------------------------------
    # Initialization: regret construction (same as parent)
    # ------------------------------------------------------------
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))

    def best_insertion(c, routes, route_dists):
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            other_max = 0.0
            for j, d in enumerate(route_dists):
                if j != r_idx and d > other_max:
                    other_max = d
            for pos in range(1, len(route)):
                pred = route[pos-1]
                succ = route[pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    # Regret construction
    while unassigned:
        bests = []
        for c in unassigned:
            best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
            bests.append((-regret, c, best_route, best_pos, best_new_max))
        bests.sort(key=lambda x: (x[0], x[1]))
        _, c, best_route, best_pos, new_max = bests[0]
        route = routes[best_route]
        route.insert(best_pos, c)
        route_dists[best_route] = route_dist(route)
        unassigned.remove(c)

    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    current_routes = [route[:] for route in routes]
    current_dists = route_dists[:]

    # ------------------------------------------------------------
    # LNS phase (reduced iterations)
    # ------------------------------------------------------------
    max_lns_iter = n  # instead of n*10
    for _ in range(max_lns_iter):
        num_remove = random.randint(max(1, (n-1)//10), max(1, (n-1)*4//10))
        customers = list(range(1, n))
        random.shuffle(customers)
        to_remove = customers[:num_remove]
        temp_routes = [route[:] for route in current_routes]
        temp_dists = current_dists[:]
        for c in to_remove:
            for r_idx in range(truck_count):
                if c in temp_routes[r_idx]:
                    pos = temp_routes[r_idx].index(c)
                    pred = temp_routes[r_idx][pos-1]
                    succ = temp_routes[r_idx][pos+1]
                    temp_dists[r_idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                    temp_routes[r_idx].pop(pos)
                    break
        unassigned = to_remove[:]
        while unassigned:
            bests = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion(c, temp_routes, temp_dists)
                if best_route == -1:
                    continue
                regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                bests.append((-regret, c, best_route, best_pos, best_new_max))
            bests.sort(key=lambda x: (x[0], x[1]))
            _, c, best_route, best_pos, new_max = bests[0]
            route = temp_routes[best_route]
            route.insert(best_pos, c)
            temp_dists[best_route] = route_dist(route)
            unassigned.remove(c)
        new_max = max(temp_dists)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [route[:] for route in temp_routes]
            report_best_vrp(best_routes)
            current_routes = [route[:] for route in temp_routes]
            current_dists = temp_dists[:]

    # ------------------------------------------------------------
    # Intensification: local search to reduce max distance
    # ------------------------------------------------------------
    max_local_iter = n * 10  # bounded
    for _ in range(max_local_iter):
        # Find the route with maximum distance
        max_idx = 0
        for i in range(truck_count):
            if route_dists[i] > route_dists[max_idx]:
                max_idx = i
        longest_route = current_routes[max_idx]
        if len(longest_route) <= 2:  # only depot
            break
        improved = False
        # Iterate over customers in the longest route (except depots)
        for c in longest_route[1:-1]:  # skip first and last (depot 0)
            # Temporarily remove c
            pos_c = longest_route.index(c)
            pred_c = longest_route[pos_c-1]
            succ_c = longest_route[pos_c+1]
            new_dist_long = route_dists[max_idx] - distance_matrix[pred_c, c] - distance_matrix[c, succ_c] + distance_matrix[pred_c, succ_c]
            other_indices = [i for i in range(truck_count) if i != max_idx]
            # Try insert into other routes
            best_new_max = float('inf')
            best_route_idx = -1
            best_pos = -1
            for r_idx in other_indices:
                other_route = current_routes[r_idx]
                other_dist = route_dists[r_idx]
                for pos in range(1, len(other_route)):
                    pred = other_route[pos-1]
                    succ = other_route[pos]
                    new_dist_other = other_dist - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                    new_max_candidate = new_dist_long
                    for k in range(truck_count):
                        if k == max_idx:
                            cand = new_dist_long
                        elif k == r_idx:
                            cand = new_dist_other
                        else:
                            cand = route_dists[k]
                        if cand > new_max_candidate:
                            new_max_candidate = cand
                    if new_max_candidate < best_new_max:
                        best_new_max = new_max_candidate
                        best_route_idx = r_idx
                        best_pos = pos
            if best_new_max < best_max - 1e-12:
                # Perform move
                # Remove c from longest route
                longest_route.pop(pos_c)
                route_dists[max_idx] = new_dist_long
                # Insert into best route
                other_route = current_routes[best_route_idx]
                other_route.insert(best_pos, c)
                route_dists[best_route_idx] = route_dist(other_route)
                best_max = max(route_dists)
                best_routes = [route[:] for route in current_routes]
                report_best_vrp(best_routes)
                improved = True
                break  # restart search after a successful move
        if not improved:
            break  # local optimum

    return best_routes