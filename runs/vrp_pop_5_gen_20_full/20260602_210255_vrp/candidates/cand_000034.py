import numpy as np
import random
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
    
    def greedy_construction(shuffled_customers):
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(shuffled_customers)
        route_dists = [0.0 for _ in range(truck_count)]
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
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
                    elif abs(new_max - best_new_max) < 1e-12 and r_idx < best_route_idx:
                        best_new_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
            routes[best_route_idx].insert(best_pos, cust)
            route_dists[best_route_idx] = compute_route_dist(routes[best_route_idx])
        return routes, route_dists
    
    def local_search(routes, route_dists, max_iter):
        best_max = max(route_dists)
        for _ in range(max_iter):
            current_max = max(route_dists)
            max_route_idxs = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
            improved = False
            # relocate from max route
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
                                if max(route_dists) < best_max - 1e-12:
                                    best_max = max(route_dists)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # swap between max and another
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
                                if max(route_dists) < best_max - 1e-12:
                                    best_max = max(route_dists)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt on max routes
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
                            if max(route_dists) < best_max - 1e-12:
                                best_max = max(route_dists)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # cross exchange between max and another
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
                                if max(route_dists) < best_max - 1e-12:
                                    best_max = max(route_dists)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes, route_dists, best_max
    
    # Generate multiple initial solutions
    best_routes = None
    best_max = float('inf')
    for seed in range(5):
        random.seed(seed)
        customers = list(range(1, n))
        random.shuffle(customers)
        routes, route_dists = greedy_construction(customers)
        routes, route_dists, current_best = local_search(routes, route_dists, max_iter=min(n*10, 500))
        if current_best < best_max - 1e-12:
            best_max = current_best
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    # Perturb and restart on best solution
    if best_routes is None:
        best_routes = [[0,0] for _ in range(truck_count)]
        best_max = 0.0
    routes = [list(r) for r in best_routes]
    route_dists = [compute_route_dist(r) for r in routes]
    for restart in range(5):
        # random perturbation: move a random customer from a random route to another random route
        # choose a route that has customers (not depot only)
        valid_routes = [i for i, r in enumerate(routes) if len(r) > 2]
        if len(valid_routes) < 2:
            break
        r_idx = random.choice(valid_routes)
        route = routes[r_idx]
        customers_in_route = [node for node in route if node != 0]
        if not customers_in_route:
            continue
        cust = random.choice(customers_in_route)
        pos = route.index(cust)
        # remove from current route
        route.pop(pos)
        route_dists[r_idx] = compute_route_dist(route)
        # choose another route (could be same? avoid self)
        other_idxs = [i for i in range(truck_count) if i != r_idx and len(routes[i]) > 1]  # at least depot
        if not other_idxs:
            # insert back to original
            route.insert(pos, cust)
            route_dists[r_idx] = compute_route_dist(route)
            continue
        new_r_idx = random.choice(other_idxs)
        # insert at best position to minimize its distance increase
        best_increase = float('inf')
        best_pos = -1
        for insert_pos in range(1, len(routes[new_r_idx])):
            prev2 = routes[new_r_idx][insert_pos-1]
            succ2 = routes[new_r_idx][insert_pos]
            increase = distance_matrix[prev2][cust] + distance_matrix[cust][succ2] - distance_matrix[prev2][succ2]
            if increase < best_increase - 1e-12:
                best_increase = increase
                best_pos = insert_pos
        routes[new_r_idx].insert(best_pos, cust)
        route_dists[new_r_idx] = compute_route_dist(routes[new_r_idx])
        # run local search
        routes, route_dists, new_best = local_search(routes, route_dists, max_iter=min(n*10, 500))
        if new_best < best_max - 1e-12:
            best_max = new_best
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
    # Ensure each route starts and ends at 0
    for r in best_routes:
        if r[0] != 0:
            r.insert(0, 0)
        if r[-1] != 0:
            r.append(0)
    return best_routes