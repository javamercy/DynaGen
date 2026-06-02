import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    # Construction: round-robin assignment + nearest neighbor
    customers = list(range(1, n))
    # sort customers by distance from depot descending, tie by index
    customers.sort(key=lambda c: (-distance_matrix[0][c], c))
    # assign to trucks round-robin
    truck_customers = [[] for _ in range(truck_count)]
    for idx, cust in enumerate(customers):
        truck_customers[idx % truck_count].append(cust)
    # build routes via nearest neighbor
    routes = [[0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for t_idx in range(truck_count):
        assigned = set(truck_customers[t_idx])
        current = 0
        while assigned:
            # find nearest unassigned customer
            best_cust = None
            best_dist = float('inf')
            for c in assigned:
                d = distance_matrix[current][c]
                if d < best_dist or (d == best_dist and c < best_cust):
                    best_dist = d
                    best_cust = c
            routes[t_idx].append(best_cust)
            assigned.remove(best_cust)
            current = best_cust
        routes[t_idx].append(0)
        route_dists[t_idx] = compute_route_distance(routes[t_idx], distance_matrix)
    # initial best
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    # improvement (same as parent)
    def compute_route_distance(route, dist):
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i]][route[i+1]]
        return d
    def apply_relocate(routes, r_idx, pos, cust, new_r_idx, new_pos):
        routes[r_idx].pop(pos)
        routes[new_r_idx].insert(new_pos, cust)
        route_dists[r_idx] = compute_route_distance(routes[r_idx], distance_matrix)
        route_dists[new_r_idx] = compute_route_distance(routes[new_r_idx], distance_matrix)
    def apply_swap(routes, r1, pos1, r2, pos2):
        cust1 = routes[r1][pos1]
        cust2 = routes[r2][pos2]
        routes[r1][pos1] = cust2
        routes[r2][pos2] = cust1
        route_dists[r1] = compute_route_distance(routes[r1], distance_matrix)
        route_dists[r2] = compute_route_distance(routes[r2], distance_matrix)
    def apply_two_opt(routes, r_idx, i, j):
        route = routes[r_idx]
        route[i:j+1] = reversed(route[i:j+1])
        route_dists[r_idx] = compute_route_distance(route, distance_matrix)
    def apply_cross(routes, r1, pos1, r2, pos2):
        tail1 = routes[r1][pos1+1:]
        tail2 = routes[r2][pos2+1:]
        new_route1 = routes[r1][:pos1+1] + tail2
        new_route2 = routes[r2][:pos2+1] + tail1
        routes[r1] = new_route1
        routes[r2] = new_route2
        route_dists[r1] = compute_route_distance(new_route1, distance_matrix)
        route_dists[r2] = compute_route_distance(new_route2, distance_matrix)
    max_iter = min(n * 20, 1000)
    for _ in range(max_iter):
        current_max = max(route_dists)
        max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
        improved = False
        # relocate
        for r_idx in max_routes:
            route = routes[r_idx]
            for pos in range(1, len(route)-1):
                cust = route[pos]
                prev = route[pos-1]
                succ = route[pos+1]
                removal_change = distance_matrix[prev][succ] - (distance_matrix[prev][cust] + distance_matrix[cust][succ])
                new_dist_r = route_dists[r_idx] + removal_change
                for other_idx in range(truck_count):
                    if other_idx == r_idx:
                        continue
                    other_route = routes[other_idx]
                    for insert_pos in range(1, len(other_route)):
                        prev2 = other_route[insert_pos-1]
                        succ2 = other_route[insert_pos]
                        insertion_change = distance_matrix[prev2][cust] + distance_matrix[cust][succ2] - distance_matrix[prev2][succ2]
                        new_dist_other = route_dists[other_idx] + insertion_change
                        new_max = max(new_dist_r, new_dist_other, max(route_dists[:r_idx]+route_dists[r_idx+1:other_idx]+route_dists[other_idx+1:]))
                        if new_max < current_max - 1e-12:
                            apply_relocate(routes, r_idx, pos, cust, other_idx, insert_pos)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            if max(route_dists) < best_max - 1e-12:
                best_max = max(route_dists)
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            continue
        # swap
        for r1 in max_routes:
            route1 = routes[r1]
            for pos1 in range(1, len(route1)-1):
                cust1 = route1[pos1]
                for r2 in range(truck_count):
                    if r2 == r1:
                        continue
                    route2 = routes[r2]
                    for pos2 in range(1, len(route2)-1):
                        cust2 = route2[pos2]
                        prev1 = route1[pos1-1]
                        succ1 = route1[pos1+1]
                        remove1_change = distance_matrix[prev1][succ1] - (distance_matrix[prev1][cust1] + distance_matrix[cust1][succ1])
                        prev2 = route2[pos2-1]
                        succ2 = route2[pos2+1]
                        remove2_change = distance_matrix[prev2][succ2] - (distance_matrix[prev2][cust2] + distance_matrix[cust2][succ2])
                        insert1_change = distance_matrix[prev1][cust2] + distance_matrix[cust2][succ1] - distance_matrix[prev1][succ1]
                        insert2_change = distance_matrix[prev2][cust1] + distance_matrix[cust1][succ2] - distance_matrix[prev2][succ2]
                        new_dist_r1 = route_dists[r1] + remove1_change + insert1_change
                        new_dist_r2 = route_dists[r2] + remove2_change + insert2_change
                        new_max = max(new_dist_r1, new_dist_r2, max(route_dists[:r1]+route_dists[r1+1:r2]+route_dists[r2+1:]))
                        if new_max < current_max - 1e-12:
                            apply_swap(routes, r1, pos1, r2, pos2)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            if max(route_dists) < best_max - 1e-12:
                best_max = max(route_dists)
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            continue
        # 2-opt
        for r_idx in max_routes:
            route = routes[r_idx]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                    new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                    change = new_edges - old_edges
                    new_dist = route_dists[r_idx] + change
                    if new_dist < current_max - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        route_dists[r_idx] = compute_route_distance(route, distance_matrix)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            if max(route_dists) < best_max - 1e-12:
                best_max = max(route_dists)
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            continue
        # cross-route exchange
        for r1 in max_routes:
            route1 = routes[r1]
            for pos1 in range(0, len(route1)-1):
                if pos1 == len(route1)-1:
                    continue
                for r2 in range(truck_count):
                    if r2 == r1:
                        continue
                    route2 = routes[r2]
                    for pos2 in range(0, len(route2)-1):
                        if pos2 == len(route2)-1:
                            continue
                        delta1 = distance_matrix[route1[pos1]][route2[pos2+1]] - distance_matrix[route1[pos1]][route1[pos1+1]]
                        delta2 = distance_matrix[route2[pos2]][route1[pos1+1]] - distance_matrix[route2[pos2]][route2[pos2+1]]
                        new_dist_r1 = route_dists[r1] + delta1
                        new_dist_r2 = route_dists[r2] + delta2
                        new_max = max(new_dist_r1, new_dist_r2, max(route_dists[:r1]+route_dists[r1+1:r2]+route_dists[r2+1:]))
                        if new_max < current_max - 1e-12:
                            apply_cross(routes, r1, pos1, r2, pos2)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            if max(route_dists) < best_max - 1e-12:
                best_max = max(route_dists)
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            continue
        break
    return best_routes