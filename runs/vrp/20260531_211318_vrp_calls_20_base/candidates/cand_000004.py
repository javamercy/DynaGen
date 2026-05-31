import numpy as np
import math
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    customers = list(range(1, n))
    m = len(customers)
    
    if truck_count >= m:
        routes = [[0, c, 0] for c in customers] + [[0, 0] for _ in range(truck_count - m)]
        return routes[:truck_count]
    
    # Step 1: Farthest-first cluster center selection
    centers = []
    # first center: farthest customer from depot
    max_dist = -1
    first_center = None
    for c in customers:
        dist = distance_matrix[0, c]
        if dist > max_dist:
            max_dist = dist
            first_center = c
        elif dist == max_dist and c < first_center:  # tie-break by index
            first_center = c
    centers.append(first_center)
    
    # distances from each customer to nearest center
    dist_to_nearest = {}
    for c in customers:
        dist_to_nearest[c] = distance_matrix[0, c] if c == first_center else distance_matrix[c, first_center]
    
    for _ in range(1, truck_count):
        # select customer with maximum distance to nearest center
        max_dist = -1
        next_center = None
        for c in customers:
            if c in centers:
                continue
            d = dist_to_nearest[c]
            if d > max_dist:
                max_dist = d
                next_center = c
            elif d == max_dist and c < next_center:
                next_center = c
        if next_center is None:
            break
        centers.append(next_center)
        # update distances
        for c in customers:
            if c in centers:
                continue
            d = distance_matrix[c, next_center]
            if d < dist_to_nearest[c]:
                dist_to_nearest[c] = d
    
    # Step 2: Assign each customer to nearest center
    clusters = defaultdict(list)
    for c in customers:
        min_dist = float('inf')
        best_center = None
        for center in centers:
            d = distance_matrix[c, center]
            if d < min_dist or (d == min_dist and center < best_center):
                min_dist = d
                best_center = center
        clusters[best_center].append(c)
    
    # Step 3: Build routes for each cluster using nearest neighbor and 2-opt
    def route_distance(route):
        if len(route) <= 2:
            return 0
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    
    def nearest_neighbor_tsp(center, cust_list):
        # start at depot 0, end at depot 0
        unvisited = set(cust_list)
        current = center
        path = [0, center]
        unvisited.discard(center)
        while unvisited:
            next_c = None
            min_dist = float('inf')
            for c in unvisited:
                d = distance_matrix[current, c]
                if d < min_dist or (d == min_dist and c < next_c):
                    min_dist = d
                    next_c = c
            path.append(next_c)
            unvisited.remove(next_c)
            current = next_c
        path.append(0)
        return path
    
    def two_opt(route):
        improved = True
        max_iter = m
        iter_count = 0
        best_route = route[:]
        best_dist = route_distance(best_route)
        while improved and iter_count < max_iter:
            improved = False
            for i in range(1, len(best_route)-2):
                for j in range(i+1, len(best_route)-1):
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dist:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
            iter_count += 1
        return best_route
    
    routes = []
    for center in centers:
        if not clusters[center]:
            routes.append([0, 0])
        else:
            route = nearest_neighbor_tsp(center, clusters[center])
            improved_route = two_opt(route)
            routes.append(improved_route)
    
    # Ensure exactly truck_count routes
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    current_routes = routes[:]
    best_routes = [r[:] for r in routes]
    best_max_dist = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)  # initial
    
    # Step 4: Improve balancing by moving customers from max route to min route
    max_iter = m * truck_count
    for _ in range(max_iter):
        # Find routes with max and min distance
        dists = [route_distance(r) for r in current_routes]
        max_idx = max(range(truck_count), key=lambda i: dists[i])
        min_idx = min(range(truck_count), key=lambda i: dists[i])
        if max_idx == min_idx or dists[max_idx] == 0:
            break
        # Attempt to move a customer from max route to min route
        max_route = current_routes[max_idx]
        min_route = current_routes[min_idx]
        # Find the best customer to move: try each customer in max_route (excluding depot)
        best_new_max = None
        best_new_min = None
        best_new_max_dist = dists[max_idx]
        best_new_min_dist = dists[min_idx]
        improvement = False
        for cust in max_route[1:-1]:
            # Remove cust from max_route
            new_max_route = [0] + [c for c in max_route[1:-1] if c != cust] + [0]
            # Insert cust into min_route at best position (min increase)
            best_pos = 1
            best_inc = float('inf')
            for pos in range(1, len(min_route)):
                inc = distance_matrix[min_route[pos-1], cust] + distance_matrix[cust, min_route[pos]] - distance_matrix[min_route[pos-1], min_route[pos]]
                if inc < best_inc:
                    best_inc = inc
                    best_pos = pos
            new_min_route = min_route[:best_pos] + [cust] + min_route[best_pos:]
            new_max_dist = route_distance(new_max_route)
            new_min_dist = route_distance(new_min_route)
            # Check if moving reduces the overall max distance
            new_overall_max = max(new_max_dist, new_min_dist, max(dists[:max_idx] + dists[max_idx+1:min_idx] + dists[min_idx+1:], default=0))
            if new_overall_max < best_max_dist or (new_overall_max == best_max_dist and (new_max_dist + new_min_dist) < (dists[max_idx] + dists[min_idx])):
                best_new_max = new_max_route
                best_new_min = new_min_route
                best_new_max_dist = new_max_dist
                best_new_min_dist = new_min_dist
                improvement = True
        if improvement:
            current_routes[max_idx] = best_new_max
            current_routes[min_idx] = best_new_min
            new_max = max(route_distance(r) for r in current_routes)
            if new_max < best_max_dist:
                best_max_dist = new_max
                best_routes = [r[:] for r in current_routes]
                report_best_vrp(best_routes)
        else:
            break
    
    # Final sanity: ensure all customers appear exactly once
    seen = set()
    for r in best_routes:
        for c in r[1:-1]:
            seen.add(c)
    for c in customers:
        if c not in seen:
            # assign to shortest route
            best_routes[-1].insert(-1, c)
    
    return best_routes