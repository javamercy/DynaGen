import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    
    # Construction: greedy insertion minimizing max route distance
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    unassigned.sort(key=lambda c: (-distance_matrix[0][c], c))
    
    def compute_route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    route_dists = [compute_route_dist(r) for r in routes]
    
    for cust in unassigned:
        best_new_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev = route[pos-1]
                succ = route[pos]
                increase = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
                new_route_dist = route_dists[r_idx] + increase
                new_max = new_route_dist
                for other_idx, d in enumerate(route_dists):
                    if other_idx != r_idx and d > new_max:
                        new_max = d
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_dists[best_route_idx] = compute_route_dist(route)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    
    # Local search helpers
    def apply_relocate(routes, r_idx, pos, cust, new_r_idx, new_pos):
        routes[r_idx].pop(pos)
        routes[new_r_idx].insert(new_pos, cust)
        route_dists[r_idx] = compute_route_dist(routes[r_idx])
        route_dists[new_r_idx] = compute_route_dist(routes[new_r_idx])
    
    def apply_swap(routes, r1, pos1, r2, pos2):
        routes[r1][pos1], routes[r2][pos2] = routes[r2][pos2], routes[r1][pos1]
        route_dists[r1] = compute_route_dist(routes[r1])
        route_dists[r2] = compute_route_dist(routes[r2])
    
    def apply_two_opt(routes, r_idx, i, j):
        route = routes[r_idx]
        route[i:j+1] = reversed(route[i:j+1])
        route_dists[r_idx] = compute_route_dist(route)
    
    def perturb(routes, route_dists):
        max_dist = max(route_dists)
        max_routes = [i for i, d in enumerate(route_dists) if abs(d - max_dist) < 1e-12]
        if not max_routes:
            return
        r_max = max_routes[0]
        route = routes[r_max]
        best_cust = None
        best_pos = None
        best_dist_from_depot = -1
        for pos in range(1, len(route)-1):
            cust = route[pos]
            d = distance_matrix[0][cust]
            if d > best_dist_from_depot:
                best_dist_from_depot = d
                best_cust = cust
                best_pos = pos
        if best_cust is None:
            return
        min_dist = min(route_dists)
        min_routes = [i for i, d in enumerate(route_dists) if abs(d - min_dist) < 1e-12]
        r_min = min_routes[0]
        best_new_max = float('inf')
        best_insert_pos = -1
        for pos in range(1, len(routes[r_min])):
            new_dist_r_max = route_dists[r_max] - (distance_matrix[route[best_pos-1]][best_cust] + distance_matrix[best_cust][route[best_pos+1]] - distance_matrix[route[best_pos-1]][route[best_pos+1]])
            new_dist_r_min = route_dists[r_min] + distance_matrix[routes[r_min][pos-1]][best_cust] + distance_matrix[best_cust][routes[r_min][pos]] - distance_matrix[routes[r_min][pos-1]][routes[r_min][pos]]
            new_max = max(new_dist_r_max, new_dist_r_min)
            for i, d in enumerate(route_dists):
                if i != r_max and i != r_min and d > new_max:
                    new_max = d
            if new_max < best_new_max:
                best_new_max = new_max
                best_insert_pos = pos
        if best_insert_pos == -1:
            return
        del route[best_pos]
        routes[r_min].insert(best_insert_pos, best_cust)
        route_dists[r_max] = compute_route_dist(route)
        route_dists[r_min] = compute_route_dist(routes[r_min])
    
    max_restarts = 1  # total runs = initial + 1 restart = 2
    for restart in range(max_restarts + 1):
        max_iter = min(n * 10, 500)
        for _ in range(max_iter):
            current_max = max(route_dists)
            max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
            improved = False
            
            # Relocate from max routes
            for r_idx in max_routes:
                route = routes[r_idx]
                for pos in range(1, len(route)-1):
                    cust = route[pos]
                    removal_change = distance_matrix[route[pos-1]][route[pos+1]] - (distance_matrix[route[pos-1]][cust] + distance_matrix[cust][route[pos+1]])
                    new_dist_r = route_dists[r_idx] + removal_change
                    for other_idx in range(truck_count):
                        if other_idx == r_idx:
                            continue
                        other_route = routes[other_idx]
                        for insert_pos in range(1, len(other_route)):
                            insertion_change = distance_matrix[other_route[insert_pos-1]][cust] + distance_matrix[cust][other_route[insert_pos]] - distance_matrix[other_route[insert_pos-1]][other_route[insert_pos]]
                            new_dist_other = route_dists[other_idx] + insertion_change
                            new_max = max(new_dist_r, new_dist_other)
                            for idx, d in enumerate(route_dists):
                                if idx != r_idx and idx != other_idx and d > new_max:
                                    new_max = d
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
            
            # Swap involving max routes
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
                            remove1_change = distance_matrix[route1[pos1-1]][route1[pos1+1]] - (distance_matrix[route1[pos1-1]][cust1] + distance_matrix[cust1][route1[pos1+1]])
                            remove2_change = distance_matrix[route2[pos2-1]][route2[pos2+1]] - (distance_matrix[route2[pos2-1]][cust2] + distance_matrix[cust2][route2[pos2+1]])
                            insert1_change = distance_matrix[route1[pos1-1]][cust2] + distance_matrix[cust2][route1[pos1+1]] - distance_matrix[route1[pos1-1]][route1[pos1+1]]
                            insert2_change = distance_matrix[route2[pos2-1]][cust1] + distance_matrix[cust1][route2[pos2+1]] - distance_matrix[route2[pos2-1]][route2[pos2+1]]
                            new_dist_r1 = route_dists[r1] + remove1_change + insert1_change
                            new_dist_r2 = route_dists[r2] + remove2_change + insert2_change
                            new_max = max(new_dist_r1, new_dist_r2)
                            for idx, d in enumerate(route_dists):
                                if idx != r1 and idx != r2 and d > new_max:
                                    new_max = d
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
            
            # 2-opt on max routes
            for r_idx in max_routes:
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        change = new_edges - old_edges
                        new_dist = route_dists[r_idx] + change
                        if new_dist < current_max - 1e-12:
                            apply_two_opt(routes, r_idx, i, j)
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
            
            # No improvement: break inner loop
            break
        
        # After inner loop, perturb if not last restart
        if restart < max_restarts:
            perturb(routes, route_dists)
            if max(route_dists) < best_max - 1e-12:
                best_max = max(route_dists)
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
    
    return best_routes