import numpy as np
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    
    def compute_route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    # construction: greedy insertion minimizing max distance
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    unassigned.sort(key=lambda c: (-distance_matrix[0][c], c))
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
                if new_route_dist < best_new_max:
                    new_max = new_route_dist
                    for other_idx, d in enumerate(route_dists):
                        if other_idx != r_idx and d > new_max:
                            new_max = d
                    if new_max < best_new_max or (new_max == best_new_max and r_idx < best_route_idx):
                        best_new_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
                elif new_route_dist == best_new_max:
                    if r_idx < best_route_idx:
                        best_new_max = new_route_dist
                        best_route_idx = r_idx
                        best_pos = pos
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_dists[best_route_idx] = compute_route_dist(route)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    
    # helpers for moves
    def apply_relocate(routes, r_idx, pos, cust, new_r_idx, new_pos):
        routes[r_idx].pop(pos)
        routes[new_r_idx].insert(new_pos, cust)
        route_dists[r_idx] = compute_route_dist(routes[r_idx])
        route_dists[new_r_idx] = compute_route_dist(routes[new_r_idx])
    
    def apply_swap(routes, r1, pos1, r2, pos2):
        cust1 = routes[r1][pos1]
        cust2 = routes[r2][pos2]
        routes[r1][pos1] = cust2
        routes[r2][pos2] = cust1
        route_dists[r1] = compute_route_dist(routes[r1])
        route_dists[r2] = compute_route_dist(routes[r2])
    
    def apply_two_opt(routes, r_idx, i, j):
        route = routes[r_idx]
        route[i:j+1] = reversed(route[i:j+1])
        route_dists[r_idx] = compute_route_dist(route)
    
    def apply_cross(routes, r1, pos1, r2, pos2):
        tail1 = routes[r1][pos1+1:]
        tail2 = routes[r2][pos2+1:]
        new_route1 = routes[r1][:pos1+1] + tail2
        new_route2 = routes[r2][:pos2+1] + tail1
        routes[r1] = new_route1
        routes[r2] = new_route2
        route_dists[r1] = compute_route_dist(new_route1)
        route_dists[r2] = compute_route_dist(new_route2)
    
    # perturbation types
    def pert_relocate_one(routes, route_dists):
        # move highest-index customer from max route to shortest route
        current_max = max(route_dists)
        max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
        r_idx = max_routes[0]
        route = routes[r_idx]
        customers = [node for node in route if node != 0]
        if not customers:
            return
        cust = max(customers)
        pos = route.index(cust)
        route.pop(pos)
        route_dists[r_idx] = compute_route_dist(route)
        # find route with smallest distance
        min_route_idx = min(range(truck_count), key=lambda i: (route_dists[i], i))
        best_increase = float('inf')
        best_pos = -1
        for pos2 in range(1, len(routes[min_route_idx])):
            prev2 = routes[min_route_idx][pos2-1]
            succ2 = routes[min_route_idx][pos2]
            increase = distance_matrix[prev2][cust] + distance_matrix[cust][succ2] - distance_matrix[prev2][succ2]
            if increase < best_increase:
                best_increase = increase
                best_pos = pos2
        routes[min_route_idx].insert(best_pos, cust)
        route_dists[min_route_idx] = compute_route_dist(routes[min_route_idx])
    
    def pert_swap_one(routes, route_dists):
        # swap highest-index customer from max route with lowest-index customer from shortest route
        current_max = max(route_dists)
        max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
        r_max = max_routes[0]
        route_max = routes[r_max]
        customers_max = [node for node in route_max if node != 0]
        if not customers_max:
            return
        cust_max = max(customers_max)
        pos_max = route_max.index(cust_max)
        # shortest route
        min_route_idx = min(range(truck_count), key=lambda i: (route_dists[i], i))
        route_min = routes[min_route_idx]
        customers_min = [node for node in route_min if node != 0]
        if not customers_min:
            return
        cust_min = min(customers_min)
        pos_min = route_min.index(cust_min)
        apply_swap(routes, r_max, pos_max, min_route_idx, pos_min)
    
    def pert_relocate_two(routes, route_dists):
        # move two highest-index customers from max route to two shortest routes
        current_max = max(route_dists)
        max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
        r_idx = max_routes[0]
        route = routes[r_idx]
        customers = [node for node in route if node != 0]
        if len(customers) < 2:
            pert_relocate_one(routes, route_dists)
            return
        # move highest first
        cust1 = max(customers)
        pos1 = route.index(cust1)
        route.pop(pos1)
        route_dists[r_idx] = compute_route_dist(route)
        min_route1 = min(range(truck_count), key=lambda i: (route_dists[i], i))
        best_increase = float('inf')
        best_pos1 = -1
        for pos in range(1, len(routes[min_route1])):
            prev = routes[min_route1][pos-1]
            succ = routes[min_route1][pos]
            inc = distance_matrix[prev][cust1] + distance_matrix[cust1][succ] - distance_matrix[prev][succ]
            if inc < best_increase:
                best_increase = inc
                best_pos1 = pos
        routes[min_route1].insert(best_pos1, cust1)
        route_dists[min_route1] = compute_route_dist(routes[min_route1])
        # move second highest from original max route (now updated)
        customers2 = [node for node in routes[r_idx] if node != 0]
        if not customers2:
            return
        cust2 = max(customers2)
        pos2 = routes[r_idx].index(cust2)
        routes[r_idx].pop(pos2)
        route_dists[r_idx] = compute_route_dist(routes[r_idx])
        # find new shortest route (excluding the one we just added to? we can use all, but may include that route again)
        min_route2 = min(range(truck_count), key=lambda i: (route_dists[i], i))
        best_increase2 = float('inf')
        best_pos2 = -1
        for pos in range(1, len(routes[min_route2])):
            prev = routes[min_route2][pos-1]
            succ = routes[min_route2][pos]
            inc = distance_matrix[prev][cust2] + distance_matrix[cust2][succ] - distance_matrix[prev][succ]
            if inc < best_increase2:
                best_increase2 = inc
                best_pos2 = pos
        routes[min_route2].insert(best_pos2, cust2)
        route_dists[min_route2] = compute_route_dist(routes[min_route2])
    
    perturbations = [pert_relocate_one, pert_swap_one, pert_relocate_two]
    max_restarts = 3
    for restart in range(max_restarts + 1):
        if restart > 0:
            routes = [list(r) for r in best_routes]
            route_dists = [compute_route_dist(r) for r in routes]
            pert_idx = (restart - 1) % len(perturbations)
            perturbations[pert_idx](routes, route_dists)
        else:
            routes = [list(r) for r in best_routes]
            route_dists = [compute_route_dist(r) for r in routes]
        
        # improvement loop
        max_iter = min(n * 20, 1000)
        for _ in range(max_iter):
            current_max = max(route_dists)
            max_route_idxs = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
            improved = False
            # 1. relocate from max route
            for r_idx in max_route_idxs:
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
            # 2. swap between max and another
            for r1 in max_route_idxs:
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
            # 3. 2-opt on max routes
            for r_idx in max_route_idxs:
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        change = new_edges - old_edges
                        new_dist = route_dists[r_idx] + change
                        if new_dist < current_max - 1e-12:
                            route[i:j+1] = reversed(route[i:j+1])
                            route_dists[r_idx] = compute_route_dist(route)
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
            # 4. cross exchange between max and another
            for r1 in max_route_idxs:
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
                            new_max = max(new_dist_r1, new_dist_r2)
                            for idx, d in enumerate(route_dists):
                                if idx != r1 and idx != r2 and d > new_max:
                                    new_max = d
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
            # no improvement in any move
            break
        # after improvement loop, update best if needed
        if max(route_dists) < best_max - 1e-12:
            best_max = max(route_dists)
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    return best_routes