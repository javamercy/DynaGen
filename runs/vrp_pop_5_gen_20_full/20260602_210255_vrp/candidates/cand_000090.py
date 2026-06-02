import numpy as np
import random
from copy import deepcopy

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    # Greedy construction: insert each customer to minimize max route distance
    routes = [[0, 0] for _ in range(truck_count)]
    dists = [0.0] * truck_count
    unassigned = list(range(1, n))
    random.shuffle(unassigned)  # slight randomization for restarts
    for cust in unassigned:
        best_max = float('inf')
        best_route = -1
        best_pos = -1
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                prev = route[pos-1]
                succ = route[pos]
                increase = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
                new_dist = dists[r_idx] + increase
                new_max = max(new_dist, max((dists[i] for i in range(truck_count) if i != r_idx), default=0.0))
                if new_max < best_max:
                    best_max = new_max
                    best_route = r_idx
                    best_pos = pos
        routes[best_route].insert(best_pos, cust)
        dists[best_route] = route_dist(routes[best_route])
    
    best_routes = deepcopy(routes)
    best_max = max(dists)
    report_best_vrp(best_routes)
    
    def apply_relocate(r_idx, pos, new_r_idx, new_pos):
        cust = routes[r_idx].pop(pos)
        routes[new_r_idx].insert(new_pos, cust)
        dists[r_idx] = route_dist(routes[r_idx])
        dists[new_r_idx] = route_dist(routes[new_r_idx])
    
    def apply_two_opt(r_idx, i, j):
        routes[r_idx][i:j+1] = reversed(routes[r_idx][i:j+1])
        dists[r_idx] = route_dist(routes[r_idx])
    
    def local_search(max_iter):
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            current_max = max(dists)
            # relocate moves: try moving a customer from the longest route to elsewhere
            max_idxs = [i for i, d in enumerate(dists) if abs(d - current_max) < 1e-12]
            for r_idx in max_idxs:
                route = routes[r_idx]
                for pos in range(1, len(route)-1):
                    cust = route[pos]
                    prev = route[pos-1]
                    succ = route[pos+1]
                    removal_change = distance_matrix[prev][succ] - (distance_matrix[prev][cust] + distance_matrix[cust][succ])
                    new_dist_r = dists[r_idx] + removal_change
                    for other_idx in range(truck_count):
                        if other_idx == r_idx:
                            continue
                        other_route = routes[other_idx]
                        for ins_pos in range(1, len(other_route)):
                            prev2 = other_route[ins_pos-1]
                            succ2 = other_route[ins_pos]
                            insertion_change = distance_matrix[prev2][cust] + distance_matrix[cust][succ2] - distance_matrix[prev2][succ2]
                            new_dist_other = dists[other_idx] + insertion_change
                            new_max = max(new_dist_r, new_dist_other, max((dists[i] for i in range(truck_count) if i not in [r_idx, other_idx]), default=0.0))
                            if new_max < current_max - 1e-12:
                                apply_relocate(r_idx, pos, other_idx, ins_pos)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                iterations += 1
                if max(dists) < best_max - 1e-12:
                    best_max = max(dists)
                    best_routes = deepcopy(routes)
                    report_best_vrp(best_routes)
                continue
            # 2-opt moves on each route
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        change = new - old
                        if change < -1e-12:
                            new_dist = dists[r_idx] + change
                            if new_dist < current_max:
                                apply_two_opt(r_idx, i, j)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                iterations += 1
                if max(dists) < best_max - 1e-12:
                    best_max = max(dists)
                    best_routes = deepcopy(routes)
                    report_best_vrp(best_routes)
    
    def perturb():
        # move one customer from longest to shortest route
        max_idx = max(range(truck_count), key=lambda i: dists[i])
        min_idx = min(range(truck_count), key=lambda i: dists[i])
        if max_idx == min_idx or len(routes[max_idx]) <= 2:
            return
        # choose a random customer from max route (excluding depot)
        route_max = routes[max_idx]
        cust_pos = random.randint(1, len(route_max)-2)
        cust = route_max[cust_pos]
        route_max.pop(cust_pos)
        dists[max_idx] = route_dist(route_max)
        # insert into min route at random position
        route_min = routes[min_idx]
        ins_pos = random.randint(1, len(route_min)-1)
        route_min.insert(ins_pos, cust)
        dists[min_idx] = route_dist(route_min)
    
    # Main loop: multiple restarts
    for restart in range(5):
        if restart > 0:
            # reset to best solution
            routes = deepcopy(best_routes)
            dists = [route_dist(r) for r in routes]
            # perform a few random perturbations
            for _ in range(2):
                perturb()
        local_search(max_iter=n*10)
        if max(dists) < best_max - 1e-12:
            best_max = max(dists)
            best_routes = deepcopy(routes)
            report_best_vrp(best_routes)
    
    return best_routes