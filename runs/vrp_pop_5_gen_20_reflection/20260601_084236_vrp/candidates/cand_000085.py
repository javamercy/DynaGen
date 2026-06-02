import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        best = route[:]
        best_d = route_distance(route)
        max_iter = n * 5
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new)
                    if d < best_d - 1e-12:
                        best_d = d
                        best = new
                        improved = True
            route = best
            iter_count += 1
        return best

    # Initial construction: nearest neighbor to form tours, then assign to trucks
    unvisited = set(customers)
    tours = []
    while unvisited:
        curr = 0
        tour = [0]
        while True:
            nearest = None
            min_dist = float('inf')
            for c in unvisited:
                d = distance_matrix[curr, c]
                if d < min_dist:
                    min_dist = d
                    nearest = c
            if nearest is None:
                break
            tour.append(nearest)
            unvisited.remove(nearest)
            curr = nearest
        tour.append(0)
        tours.append(tour)
    
    # distribute tours evenly to trucks
    # if fewer tours than trucks, split longest tours
    while len(tours) < truck_count:
        # split the longest tour into two
        longest_idx = max(range(len(tours)), key=lambda i: route_distance(tours[i]))
        longest = tours[longest_idx]
        if len(longest) <= 3:
            break
        # find best split point
        best_split = 1
        best_sum = float('inf')
        for split in range(1, len(longest)-1):
            tour1 = longest[:split+1] + [0]
            tour2 = [0] + longest[split:]
            d1 = route_distance(tour1)
            d2 = route_distance(tour2)
            if max(d1, d2) < best_sum:
                best_sum = max(d1, d2)
                best_split = split
        tour1 = longest[:best_split+1] + [0]
        tour2 = [0] + longest[best_split:]
        tours[longest_idx] = tour1
        tours.append(tour2)
    
    # if more tours than trucks, combine shortest tours
    while len(tours) > truck_count:
        # find two shortest tours and merge
        tours.sort(key=route_distance)
        t1 = tours.pop(0)
        t2 = tours.pop(0)
        # merge t1 and t2 by concatenating without depot in between
        merged = t1[:-1] + t2[1:]
        merged = two_opt(merged)
        tours.append(merged)
    
    # assign to routes list
    routes = tours[:]
    # apply 2-opt to each
    for t in range(truck_count):
        routes[t] = two_opt(routes[t])
    best_routes = [r[:] for r in routes]
    best_max = max_distance(routes)
    report_best_vrp(best_routes)
    cur_max = best_max
    cur_total = total_distance(routes)

    # Inter-route 2-opt* best improvement (bounded iterations)
    improved = True
    max_iter_improve = max(n, 10)
    iter_count = 0
    while improved and iter_count < max_iter_improve:
        improved = False
        best_improv = None
        best_new_max = float('inf')
        best_new_total = float('inf')
        for t1 in range(truck_count):
            for t2 in range(t1+1, truck_count):
                r1 = routes[t1]
                r2 = routes[t2]
                if len(r1) <= 2 or len(r2) <= 2:
                    continue
                for i in range(1, len(r1)-1):
                    for j in range(1, len(r2)-1):
                        new_r1 = r1[:i+1] + r2[j+1:]
                        new_r2 = r2[:j+1] + r1[i+1:]
                        d1 = route_distance(new_r1)
                        d2 = route_distance(new_r2)
                        other_max = 0.0
                        other_total = 0.0
                        for idx, r in enumerate(routes):
                            if idx not in (t1, t2):
                                d = route_distance(r)
                                if d > other_max:
                                    other_max = d
                                other_total += d
                        cand_max = max(d1, d2, other_max)
                        cand_total = d1 + d2 + other_total
                        if cand_max < best_new_max - 1e-12 or (abs(cand_max - best_new_max) < 1e-12 and cand_total < best_new_total - 1e-12):
                            best_new_max = cand_max
                            best_new_total = cand_total
                            best_improv = (t1, t2, i, j, new_r1, new_r2)
        if best_improv is not None and (best_new_max < cur_max - 1e-12 or (abs(best_new_max - cur_max) < 1e-12 and best_new_total < cur_total - 1e-12)):
            t1, t2, i, j, new_r1, new_r2 = best_improv
            routes[t1] = two_opt(new_r1)
            routes[t2] = two_opt(new_r2)
            cur_max = max_distance(routes)
            cur_total = total_distance(routes)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
            improved = True
            iter_count += 1

    # Max-route reduction via relocation (bounded attempts)
    reduction_attempts = 0
    while reduction_attempts < n:
        max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
        max_route = routes[max_idx]
        if len(max_route) <= 2:
            break
        found = False
        for idx in range(1, len(max_route)-1):
            cust = max_route[idx]
            new_max_route = max_route[:idx] + max_route[idx+1:]
            for t2 in range(truck_count):
                if t2 == max_idx:
                    continue
                r2 = routes[t2]
                for pos in range(1, len(r2)):
                    new_r2 = r2[:pos] + [cust] + r2[pos:]
                    d_max_new = route_distance(new_max_route)
                    d2_new = route_distance(new_r2)
                    other_max = 0.0
                    other_total = 0.0
                    for idx2, r in enumerate(routes):
                        if idx2 not in (max_idx, t2):
                            d = route_distance(r)
                            if d > other_max:
                                other_max = d
                            other_total += d
                    cand_max = max(d_max_new, d2_new, other_max)
                    cand_total = d_max_new + d2_new + other_total
                    if cand_max < cur_max - 1e-12 or (abs(cand_max - cur_max) < 1e-12 and cand_total < cur_total - 1e-12):
                        routes[max_idx] = two_opt(new_max_route)
                        routes[t2] = two_opt(new_r2)
                        cur_max = max_distance(routes)
                        cur_total = total_distance(routes)
                        if cur_max < best_max - 1e-12:
                            best_max = cur_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            reduction_attempts += 1
        else:
            break

    # Final 2-opt
    if best_routes:
        for t in range(truck_count):
            best_routes[t] = two_opt(best_routes[t])
        best_max = max_distance(best_routes)
        report_best_vrp(best_routes)
    return best_routes