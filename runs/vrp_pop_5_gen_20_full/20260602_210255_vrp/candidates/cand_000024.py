import numpy as np
from collections import defaultdict
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    # construction
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    unassigned.sort(key=lambda c: (-distance_matrix[0][c], c))
    def route_distance(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i]][route[i+1]]
        return total
    route_dists = [route_distance(r) for r in routes]
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
                        best_new_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_dists[best_route_idx] = route_distance(route)
    # initial best
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    
    # improvement functions
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
        route_dists[r1] = compute_route_dist(routes[r1])
        route_dists[r2] = compute_route_dist(routes[r2])
    
    def improve(routes, route_dists):
        improved = True
        while improved:
            improved = False
            # relocate
            for r_idx in range(truck_count):
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
                            if new_max < max(route_dists) - 1e-12:
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
                continue
            # swap
            for r1 in range(truck_count):
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
                            if new_max < max(route_dists) - 1e-12:
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
                continue
            # 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        change = new_edges - old_edges
                        new_dist = route_dists[r_idx] + change
                        if new_dist < max(route_dists) - 1e-12:
                            apply_two_opt(routes, r_idx, i, j)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # cross
            for r1 in range(truck_count):
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
                            if new_max < max(route_dists) - 1e-12:
                                apply_cross(routes, r1, pos1, r2, pos2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes, route_dists
    
    # initial improvement
    routes, route_dists = improve(routes, route_dists)
    if max(route_dists) < best_max - 1e-12:
        best_max = max(route_dists)
        best_routes = [list(r) for r in routes]
        report_best_vrp(best_routes)
    
    # restarts with perturbation
    max_restarts = min(n, 10)
    for restart in range(max_restarts):
        # perturbation: random relocations of up to 5 customers from max-cost routes
        perturbed_routes = [list(r) for r in best_routes]
        pert_dists = [compute_route_dist(r) for r in perturbed_routes]
        max_dist = max(pert_dists)
        candidates = [i for i, d in enumerate(pert_dists) if abs(d - max_dist) < 1e-12]
        num_moves = min(5, len(candidates))
        for _ in range(num_moves):
            r_idx = random.choice(candidates)
            route = perturbed_routes[r_idx]
            if len(route) <= 2:
                continue
            pos = random.randint(1, len(route)-2)
            cust = route[pos]
            # remove cust
            route.pop(pos)
            # insert into a random different route at random position
            other_idx = random.choice([i for i in range(truck_count) if i != r_idx])
            other_route = perturbed_routes[other_idx]
            insert_pos = random.randint(1, len(other_route)-1)
            other_route.insert(insert_pos, cust)
            # update distances for both routes
            pert_dists[r_idx] = compute_route_dist(route)
            pert_dists[other_idx] = compute_route_dist(other_route)
        # improve perturbed solution
        perturbed_routes, pert_dists = improve(perturbed_routes, pert_dists)
        current_max = max(pert_dists)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [list(r) for r in perturbed_routes]
            report_best_vrp(best_routes)
        # else, no improvement from this restart
    return best_routes