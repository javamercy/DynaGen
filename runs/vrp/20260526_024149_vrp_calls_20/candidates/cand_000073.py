import numpy as np
import random
import math

def route_distance(route, dm):
    return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))

def construct_greedy(customers, truck_count, dm):
    routes = [[0, 0] for _ in range(truck_count)]
    for cust in customers:
        best_truck = None
        best_pos = None
        best_new_max = float('inf')
        for t in range(truck_count):
            route = routes[t]
            current_dists = [route_distance(routes[tt], dm) for tt in range(truck_count) if tt != t]
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance(new_route, dm)
                new_max = max(max(current_dists) if current_dists else 0, new_dist)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_truck = t
                    best_pos = pos
        if best_truck is None:
            best_truck = 0
            best_pos = 1
        route = routes[best_truck]
        routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
    return routes

def local_search(routes, dm, truck_count, customers):
    n = len(customers)
    max_iter = min(100, n * 2)
    max_perturb = 5
    for iteration in range(max_iter):
        improved = False
        current_max = max(route_distance(r, dm) for r in routes)
        # Inter-route relocate
        for t_from in range(truck_count):
            route_from = routes[t_from]
            if len(route_from) <= 2:
                continue
            for idx_from in range(1, len(route_from)-1):
                cust = route_from[idx_from]
                new_route_from = route_from[:idx_from] + route_from[idx_from+1:]
                if len(new_route_from) == 2 and len(new_route_from[1]) == 1:
                    continue
                dist_from = route_distance(new_route_from, dm)
                for t_to in range(truck_count):
                    if t_to == t_from:
                        continue
                    route_to = routes[t_to]
                    for pos_to in range(1, len(route_to)):
                        new_route_to = route_to[:pos_to] + [cust] + route_to[pos_to:]
                        dist_to = route_distance(new_route_to, dm)
                        other_dists = [route_distance(routes[tt], dm) for tt in range(truck_count) if tt not in (t_from, t_to)]
                        new_max = max(other_dists + [dist_from, dist_to])
                        if new_max < current_max - 1e-12:
                            routes[t_from] = new_route_from
                            routes[t_to] = new_route_to
                            current_max = new_max
                            improved = True
                            report_best_vrp([r[:] for r in routes])
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Inter-route swap
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust1 = route1[idx1]
                for t2 in range(t1+1, truck_count):
                    route2 = routes[t2]
                    if len(route2) <= 2:
                        continue
                    for idx2 in range(1, len(route2)-1):
                        cust2 = route2[idx2]
                        new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                        new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                        dist1 = route_distance(new_route1, dm)
                        dist2 = route_distance(new_route2, dm)
                        other_dists = [route_distance(routes[tt], dm) for tt in range(truck_count) if tt not in (t1, t2)]
                        new_max = max(other_dists + [dist1, dist2])
                        if new_max < current_max - 1e-12:
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            current_max = new_max
                            improved = True
                            report_best_vrp([r[:] for r in routes])
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Inter-route 2-opt*: swap edges between two routes
        for t1 in range(truck_count):
            route1 = routes[t1]
            if len(route1) <= 3:
                continue
            for t2 in range(t1+1, truck_count):
                route2 = routes[t2]
                if len(route2) <= 3:
                    continue
                for i1 in range(1, len(route1)-1):
                    for i2 in range(1, len(route2)-1):
                        # new routes: route1[0:i1+1] + route2[i2:0:-1] (or something)
                        # Actually 2-opt*: reverse the segment after the cut in one route and attach to the other
                        # More standard: try all combinations of splitting each route into two parts and recombine.
                        # We'll implement a simple version: remove edges (i1,i1+1) and (i2,i2+1) and reconnect
                        new_route1 = route1[:i1+1] + route2[i2+1:]
                        new_route2 = route2[:i2+1] + route1[i1+1:]
                        dist1 = route_distance(new_route1, dm)
                        dist2 = route_distance(new_route2, dm)
                        other_dists = [route_distance(routes[tt], dm) for tt in range(truck_count) if tt not in (t1, t2)]
                        new_max = max(other_dists + [dist1, dist2])
                        if new_max < current_max - 1e-12:
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            current_max = new_max
                            improved = True
                            report_best_vrp([r[:] for r in routes])
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Intra-route 2-opt on longest route
        max_route_idx = max(range(truck_count), key=lambda t: route_distance(routes[t], dm))
        route = routes[max_route_idx]
        best_improvement = 0
        best_pair = None
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                old_dist = route_distance(route, dm)
                new_dist = route_distance(new_route, dm)
                if old_dist - new_dist > best_improvement:
                    best_improvement = old_dist - new_dist
                    best_pair = (i, j)
        if best_improvement > 1e-12:
            i, j = best_pair
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            routes[max_route_idx] = new_route
            current_max = max(route_distance(r, dm) for r in routes)
            improved = True
            report_best_vrp([r[:] for r in routes])
        if improved:
            continue
        # Perturbation: if no improvement, randomly eject 1-3 customers and reinsert greedily
        if iteration < max_iter - 1:
            # Choose a random truck to remove from (prefer longest)
            truck_idx = random.randrange(truck_count)
            route = routes[truck_idx]
            if len(route) <= 3:
                continue
            # Remove up to 3 customers
            remove_count = random.randint(1, min(3, len(route)-2))
            remove_indices = random.sample(range(1, len(route)-1), remove_count)
            remove_custs = [route[i] for i in remove_indices]
            new_route = [route[i] for i in range(len(route)) if i not in remove_indices]
            routes[truck_idx] = new_route
            # Reinsert greedily
            for cust in remove_custs:
                best_truck = None
                best_pos = None
                best_new_max = float('inf')
                for t in range(truck_count):
                    r = routes[t]
                    current_dists = [route_distance(routes[tt], dm) for tt in range(truck_count) if tt != t]
                    for pos in range(1, len(r)):
                        new_r = r[:pos] + [cust] + r[pos:]
                        new_dist = route_distance(new_r, dm)
                        new_max = max(max(current_dists) if current_dists else 0, new_dist)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_truck = t
                            best_pos = pos
                if best_truck is None:
                    best_truck = 0
                    best_pos = 1
                r = routes[best_truck]
                routes[best_truck] = r[:best_pos] + [cust] + r[best_pos:]
            improved = True  # force continue loop
        if not improved:
            break
    return routes

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    num_cust = n - 1
    if truck_count >= num_cust:
        routes = [[0, c, 0] for c in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes
    best_overall = None
    best_max = float('inf')
    num_restarts = max(3, min(10, num_cust // 10))
    for restart in range(num_restarts):
        random.seed(restart)
        shuffled = customers[:]
        random.shuffle(shuffled)
        routes = construct_greedy(shuffled, truck_count, distance_matrix)
        routes = local_search(routes, distance_matrix, truck_count, customers)
        current_max = max(route_distance(r, distance_matrix) for r in routes)
        if current_max < best_max:
            best_max = current_max
            best_overall = [r[:] for r in routes]
            report_best_vrp(best_overall)
    if best_overall is None:
        best_overall = routes
    return best_overall