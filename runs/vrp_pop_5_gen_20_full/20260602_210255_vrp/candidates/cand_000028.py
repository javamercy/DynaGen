import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    # construction: greedy insertion minimizing max route distance
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    unassigned.sort(key=lambda c: (-distance_matrix[0][c], c))
    def route_dist(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    route_dists = [route_dist(r) for r in routes]
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
        route_dists[best_route_idx] = route_dist(route)
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    
    # incremental update helpers
    def update_route_dist(route, old_dist, change):
        return old_dist + change
    
    def apply_relocate(routes, r_idx, pos, cust, new_r_idx, new_pos):
        # remove from old
        old_route = routes[r_idx]
        prev_old = old_route[pos-1]
        succ_old = old_route[pos+1]
        change_old = distance_matrix[prev_old][succ_old] - (distance_matrix[prev_old][cust] + distance_matrix[cust][succ_old])
        routes[r_idx].pop(pos)
        route_dists[r_idx] = update_route_dist(old_route, route_dists[r_idx], change_old)
        # insert into new
        new_route = routes[new_r_idx]
        prev_new = new_route[new_pos-1]
        succ_new = new_route[new_pos]
        change_new = distance_matrix[prev_new][cust] + distance_matrix[cust][succ_new] - distance_matrix[prev_new][succ_new]
        routes[new_r_idx].insert(new_pos, cust)
        route_dists[new_r_idx] = update_route_dist(new_route, route_dists[new_r_idx], change_new)
    
    def apply_swap(routes, r1, pos1, r2, pos2):
        cust1 = routes[r1][pos1]
        cust2 = routes[r2][pos2]
        # compute changes
        route1 = routes[r1]
        route2 = routes[r2]
        prev1 = route1[pos1-1]
        succ1 = route1[pos1+1]
        prev2 = route2[pos2-1]
        succ2 = route2[pos2+1]
        change1 = (distance_matrix[prev1][cust2] + distance_matrix[cust2][succ1] - (distance_matrix[prev1][cust1] + distance_matrix[cust1][succ1]))
        change2 = (distance_matrix[prev2][cust1] + distance_matrix[cust1][succ2] - (distance_matrix[prev2][cust2] + distance_matrix[cust2][succ2]))
        # swap
        routes[r1][pos1], routes[r2][pos2] = cust2, cust1
        route_dists[r1] = update_route_dist(route1, route_dists[r1], change1)
        route_dists[r2] = update_route_dist(route2, route_dists[r2], change2)
    
    def apply_two_opt(routes, r_idx, i, j):
        route = routes[r_idx]
        old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
        new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
        change = new_edges - old_edges
        route[i:j+1] = reversed(route[i:j+1])
        route_dists[r_idx] = update_route_dist(route, route_dists[r_idx], change)
    
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
        # compute new dist for r_max after removal
        prev = route[best_pos-1]
        succ = route[best_pos+1]
        removal_change = distance_matrix[prev][succ] - (distance_matrix[prev][best_cust] + distance_matrix[best_cust][succ])
        new_dist_r_max = route_dists[r_max] + removal_change
        for pos in range(1, len(routes[r_min])):
            prev2 = routes[r_min][pos-1]
            succ2 = routes[r_min][pos]
            insertion_change = distance_matrix[prev2][best_cust] + distance_matrix[best_cust][succ2] - distance_matrix[prev2][succ2]
            new_dist_r_min = route_dists[r_min] + insertion_change
            new_max = new_dist_r_max
            if new_dist_r_min > new_max:
                new_max = new_dist_r_min
            for idx, d in enumerate(route_dists):
                if idx != r_max and idx != r_min and d > new_max:
                    new_max = d
            if new_max < best_new_max:
                best_new_max = new_max
                best_insert_pos = pos
        if best_insert_pos == -1:
            return
        # apply move
        routes[r_max].pop(best_pos)
        routes[r_min].insert(best_insert_pos, best_cust)
        route_dists[r_max] = new_dist_r_max
        route_dists[r_min] = route_dists[r_min] + (distance_matrix[routes[r_min][best_insert_pos-1]][best_cust] + distance_matrix[best_cust][routes[r_min][best_insert_pos+1]] - distance_matrix[routes[r_min][best_insert_pos-1]][routes[r_min][best_insert_pos+1]])
        # careful: after insertion, the distances are updated; we already computed new_dist_r_min, but it's better to recompute? Actually we have the change, but positions changed.
        # Simpler: after insertion, recompute route_dists for r_min to be safe.
        route_dists[r_min] = sum(distance_matrix[routes[r_min][i]][routes[r_min][i+1]] for i in range(len(routes[r_min])-1))
    
    def local_search(routes, route_dists, max_iter):
        for _ in range(max_iter):
            current_max = max(route_dists)
            max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
            improved = False
            # relocate on max routes
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
                            new_max = new_dist_r
                            if new_dist_other > new_max:
                                new_max = new_dist_other
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
                            new_max = new_dist_r1
                            if new_dist_r2 > new_max:
                                new_max = new_dist_r2
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
            break
    
    # first improvement phase
    max_iter = min(n * 10, 500)
    local_search(routes, route_dists, max_iter)
    # perturb once
    perturb(routes, route_dists)
    if max(route_dists) < best_max - 1e-12:
        best_max = max(route_dists)
        best_routes = [list(r) for r in routes]
        report_best_vrp(best_routes)
    # second improvement phase
    local_search(routes, route_dists, max_iter)
    if max(route_dists) < best_max - 1e-12:
        best_max = max(route_dists)
        best_routes = [list(r) for r in routes]
        report_best_vrp(best_routes)
    return best_routes