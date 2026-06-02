import numpy as np
import random
from itertools import permutations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_routes_distances(routes):
        return [compute_route_distance(r) for r in routes]

    def sum_of_distances(route):
        # helper for regret
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    # Regret-2 construction
    def construct_initial():
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        route_distances = [0.0] * truck_count
        while unassigned:
            best_customer = None
            best_regret = -1.0
            best_pos = None
            for c in unassigned:
                data = []
                for r_idx, route in enumerate(routes):
                    if len(route) == 2:
                        # empty truck, can only insert at position 1
                        new_dist = distance_matrix[0, c] * 2
                        data.append((new_dist, r_idx, 1))
                    else:
                        curr_dist = route_distances[r_idx]
                        for i in range(1, len(route)):
                            new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], c] + distance_matrix[c, route[i]]
                            data.append((new_dist, r_idx, i))
                data.sort(key=lambda x: (x[0], x[1], x[2]))
                if len(data) >= 2:
                    regret = data[1][0] - data[0][0]
                else:
                    regret = 0.0
                if regret > best_regret:
                    best_regret = regret
                    best_customer = c
                    best_pos = (data[0][1], data[0][2])
                elif regret == best_regret and best_customer is not None and c < best_customer:
                    best_customer = c
                    best_pos = (data[0][1], data[0][2])
            r_idx, pos = best_pos
            route = routes[r_idx]
            route.insert(pos, best_customer)
            route_distances[r_idx] = sum_of_distances(route)
            unassigned.remove(best_customer)
        return routes, route_distances

    # Local search: intra-route 2-opt and inter-route swap
    def improve(routes, route_distances):
        improved = True
        max_passes = 10  # finite bound
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            # Intra-route 2-opt (first improvement)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = sum_of_distances(new_route)
                        if new_dist < route_distances[r_idx]:
                            route_distances[r_idx] = new_dist
                            routes[r_idx] = new_route
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                continue
            # Inter-route swap (first improvement)
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = sum_of_distances(new1)
                            new_dist2 = sum_of_distances(new2)
                            other_max = max(route_distances[:r1] + route_distances[r1+1:r2] + route_distances[r2+1:], default=0.0)
                            new_max = max(new_dist1, new_dist2, other_max)
                            if new_max < best_max_global:
                                routes[r1] = new1
                                routes[r2] = new2
                                route_distances[r1] = new_dist1
                                route_distances[r2] = new_dist2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes, route_distances

    # Build initial solution
    routes, route_distances = construct_initial()
    best_routes = [list(r) for r in routes]
    best_max = max(route_distances)
    
    def report_best_vrp(routes, route_distances):
        nonlocal best_routes, best_max
        current_max = max(route_distances)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]

    # Initial improvement
    routes, route_distances = improve(routes, route_distances)
    current_max = max(route_distances)
    if current_max < best_max:
        best_max = current_max
        best_routes = [list(r) for r in routes]

    # Iterated local search with ruin and recreate
    max_iterations = 10  # finite bound
    for iteration in range(max_iterations):
        # Ruin: remove customers from longest routes
        longest_routes = sorted(range(truck_count), key=lambda i: route_distances[i], reverse=True)
        num_to_ruin = min(3, truck_count)  # perturb up to 3 longest routes
        removed = []
        for r_idx in longest_routes[:num_to_ruin]:
            route = routes[r_idx]
            if len(route) <= 3:  # only depot and one customer? skip if too short
                continue
            # remove a random fraction (20%) of customers (excluding depot)
            customers = route[1:-1]
            k = max(1, int(len(customers) * 0.2))
            random.shuffle(customers)
            to_remove = customers[:k]
            for c in to_remove:
                route.remove(c)
                removed.append(c)
            # update route distance
            route_distances[r_idx] = sum_of_distances(route)
        if not removed:
            continue
        # Reinsert: regret-2 on current routes
        unassigned = set(removed)
        while unassigned:
            best_customer = None
            best_regret = -1.0
            best_pos = None
            for c in unassigned:
                data = []
                for r_idx, route in enumerate(routes):
                    if len(route) == 2:
                        new_dist = distance_matrix[0, c] * 2
                        data.append((new_dist, r_idx, 1))
                    else:
                        curr_dist = route_distances[r_idx]
                        for i in range(1, len(route)):
                            new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], c] + distance_matrix[c, route[i]]
                            data.append((new_dist, r_idx, i))
                data.sort(key=lambda x: (x[0], x[1], x[2]))
                if len(data) >= 2:
                    regret = data[1][0] - data[0][0]
                else:
                    regret = 0.0
                if regret > best_regret:
                    best_regret = regret
                    best_customer = c
                    best_pos = (data[0][1], data[0][2])
                elif regret == best_regret and best_customer is not None and c < best_customer:
                    best_customer = c
                    best_pos = (data[0][1], data[0][2])
            r_idx, pos = best_pos
            route = routes[r_idx]
            route.insert(pos, best_customer)
            route_distances[r_idx] = sum_of_distances(route)
            unassigned.remove(best_customer)
        # Local search after reinsertion
        routes, route_distances = improve(routes, route_distances)
        report_best_vrp(routes, route_distances)
    return best_routes