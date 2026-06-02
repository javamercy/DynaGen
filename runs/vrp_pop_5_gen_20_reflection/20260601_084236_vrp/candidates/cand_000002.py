import numpy as np
import math
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    
    # Greedy insertion: repeatedly pick customer that minimizes max route distance after insertion
    while unassigned:
        best_cost = float('inf')
        best_customer = None
        best_route_idx = None
        best_pos = None
        for cust in unassigned:
            for r_idx in range(truck_count):
                route = routes[r_idx]
                # if route is empty truck ([0,0]) then only one possible position: between 0 and 0
                positions = range(1, len(route))  # insert after index 0, before last
                for pos in positions:
                    # compute new distance for this route
                    old_dist = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = sum(distance_matrix[new_route[i], new_route[i+1]] for i in range(len(new_route)-1))
                    # compute max distance after hypothetical insertion
                    other_dists = [sum(distance_matrix[routes[j][i], routes[j][i+1]] for i in range(len(routes[j])-1)) for j in range(truck_count) if j != r_idx]
                    candidate_max = max(other_dists + [new_dist])
                    # tie-breaker: also consider increase in max (or total distance increase)
                    total_increase = new_dist - old_dist
                    # Priority: lower candidate_max, then lower total_increase, then lower customer index
                    # Use tuple for deterministic comparison
                    candidate = (candidate_max, total_increase, cust, r_idx, pos)
                    if candidate < (best_cost, best_total_increase, best_customer, best_route_idx, best_pos):
                        best_cost = candidate[0]
                        best_total_increase = candidate[1]
                        best_customer = cust
                        best_route_idx = r_idx
                        best_pos = pos
        # Apply best insertion
        route = routes[best_route_idx]
        routes[best_route_idx] = route[:best_pos] + [best_customer] + route[best_pos:]
        unassigned.remove(best_customer)
    
    # Compute initial max distance
    def route_dist(r):
        return sum(distance_matrix[r[i], r[i+1]] for i in range(len(r)-1))
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in best_routes)
    report_best_vrp(best_routes)
    
    # Improvement loop: relocate, swap, 2-opt
    improved = True
    max_iter = n * truck_count  # finite bound
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        # Relocate: move a customer from one route to another
        for r_idx1 in range(truck_count):
            r1 = routes[r_idx1]
            if len(r1) <= 2:
                continue
            for pos1 in range(1, len(r1)-1):
                cust = r1[pos1]
                # Temporarily remove cust
                r1_new = r1[:pos1] + r1[pos1+1:]
                dist1_no_cust = sum(distance_matrix[r1_new[i], r1_new[i+1]] for i in range(len(r1_new)-1))
                for r_idx2 in range(truck_count):
                    if r_idx2 == r_idx1:
                        continue
                    r2 = routes[r_idx2]
                    for pos2 in range(1, len(r2)):
                        r2_new = r2[:pos2] + [cust] + r2[pos2:]
                        dist2_new = sum(distance_matrix[r2_new[i], r2_new[i+1]] for i in range(len(r2_new)-1))
                        # Compute new max
                        new_dists = []
                        for j in range(truck_count):
                            if j == r_idx1:
                                new_dists.append(dist1_no_cust)
                            elif j == r_idx2:
                                new_dists.append(dist2_new)
                            else:
                                new_dists.append(route_dist(routes[j]))
                        new_max = max(new_dists)
                        # Improvement if strict decrease in max, or same max but decrease in total distance
                        old_max = max(route_dist(r) for r in routes)
                        if new_max < old_max or (new_max == old_max and sum(new_dists) < sum(route_dist(r) for r in routes)):
                            # Apply move
                            routes[r_idx1] = r1_new
                            routes[r_idx2] = r2_new
                            improved = True
                            # Update best if max reduced
                            if new_max < best_max:
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Swap: exchange customers between two routes
        for r_idx1, r_idx2 in combinations(range(truck_count), 2):
            r1 = routes[r_idx1]
            r2 = routes[r_idx2]
            if len(r1) <= 2 or len(r2) <= 2:
                continue
            for pos1 in range(1, len(r1)-1):
                cust1 = r1[pos1]
                for pos2 in range(1, len(r2)-1):
                    cust2 = r2[pos2]
                    # Swap
                    r1_new = r1[:pos1] + [cust2] + r1[pos1+1:]
                    r2_new = r2[:pos2] + [cust1] + r2[pos2+1:]
                    dist1_new = sum(distance_matrix[r1_new[i], r1_new[i+1]] for i in range(len(r1_new)-1))
                    dist2_new = sum(distance_matrix[r2_new[i], r2_new[i+1]] for i in range(len(r2_new)-1))
                    old_dists = [route_dist(r) for r in routes]
                    new_dists = old_dists.copy()
                    new_dists[r_idx1] = dist1_new
                    new_dists[r_idx2] = dist2_new
                    new_max = max(new_dists)
                    old_max = max(old_dists)
                    if new_max < old_max or (new_max == old_max and sum(new_dists) < sum(old_dists)):
                        routes[r_idx1] = r1_new
                        routes[r_idx2] = r2_new
                        improved = True
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            r = routes[r_idx]
            if len(r) <= 3:
                continue
            old_dist = route_dist(r)
            best_improve = 0
            best_i = best_j = None
            for i in range(1, len(r)-2):
                for j in range(i+1, len(r)-1):
                    # Reverse segment i..j
                    new_route = r[:i] + r[i:j+1][::-1] + r[j+1:]
                    new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                    if new_dist < old_dist:
                        # Improvement in this route reduces max only if this route was the max or if reduction helps
                        # We'll accept if it improves total distance (since intra-route 2-opt never increases max if this route is not the only max? Actually it could reduce max if this route was max)
                        # Accept if it reduces the current max distance
                        new_dists = [route_dist(r) for r in routes]
                        new_dists[r_idx] = new_dist
                        new_max = max(new_dists)
                        old_max = max(route_dist(r) for r in routes)
                        if new_max < old_max or (new_max == old_max and sum(new_dists) < sum(route_dist(r) for r in routes)):
                            routes[r_idx] = new_route
                            improved = True
                            if new_max < best_max:
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                            break
                if improved:
                    break
            if improved:
                break
    # Return best found routes
    return best_routes