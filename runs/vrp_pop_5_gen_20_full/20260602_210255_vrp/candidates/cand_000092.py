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

    # Construction: greedy insertion minimizing max route distance
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
                new_max = new_route_dist
                for other_idx, d in enumerate(route_dists):
                    if other_idx != r_idx and d > new_max:
                        new_max = d
                if new_max < best_new_max or (new_max == best_new_max and r_idx < best_route_idx):
                    best_new_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        route = routes[best_route_idx]
        route.insert(best_pos, cust)
        route_dists[best_route_idx] = compute_route_dist(route)

    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)

    # Helper functions for moves
    def apply_relocate(r_idx, pos, new_r_idx, new_pos):
        cust = routes[r_idx].pop(pos)
        routes[new_r_idx].insert(new_pos, cust)
        route_dists[r_idx] = compute_route_dist(routes[r_idx])
        route_dists[new_r_idx] = compute_route_dist(routes[new_r_idx])

    def apply_swap(r1, pos1, r2, pos2):
        cust1 = routes[r1][pos1]
        cust2 = routes[r2][pos2]
        routes[r1][pos1] = cust2
        routes[r2][pos2] = cust1
        route_dists[r1] = compute_route_dist(routes[r1])
        route_dists[r2] = compute_route_dist(routes[r2])

    def apply_two_opt(r_idx, i, j):
        route = routes[r_idx]
        route[i:j+1] = reversed(route[i:j+1])
        route_dists[r_idx] = compute_route_dist(route)

    def apply_cross(r1, pos1, r2, pos2):
        tail1 = routes[r1][pos1+1:]
        tail2 = routes[r2][pos2+1:]
        new_route1 = routes[r1][:pos1+1] + tail2
        new_route2 = routes[r2][:pos2+1] + tail1
        routes[r1] = new_route1
        routes[r2] = new_route2
        route_dists[r1] = compute_route_dist(new_route1)
        route_dists[r2] = compute_route_dist(new_route2)

    # First-improvement local search
    def local_search():
        nonlocal routes, route_dists, best_routes, best_max
        improved = True
        while improved:
            improved = False
            current_max = max(route_dists)
            max_route_idxs = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
            # Relocate from max routes to others
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
                                apply_relocate(r_idx, pos, other_idx, insert_pos)
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
            # Swap between max and another
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
                                apply_swap(r1, pos1, r2, pos2)
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
            for r_idx in max_route_idxs:
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        change = new_edges - old_edges
                        new_dist = route_dists[r_idx] + change
                        if new_dist < current_max - 1e-12:
                            apply_two_opt(r_idx, i, j)
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
            # Cross exchange between max and another
            for r1 in max_route_idxs:
                route1 = routes[r1]
                for pos1 in range(0, len(route1)-1):
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes[r2]
                        for pos2 in range(0, len(route2)-1):
                            delta1 = distance_matrix[route1[pos1]][route2[pos2+1]] - distance_matrix[route1[pos1]][route1[pos1+1]]
                            delta2 = distance_matrix[route2[pos2]][route1[pos1+1]] - distance_matrix[route2[pos2]][route2[pos2+1]]
                            new_dist_r1 = route_dists[r1] + delta1
                            new_dist_r2 = route_dists[r2] + delta2
                            new_max = max(new_dist_r1, new_dist_r2)
                            for idx, d in enumerate(route_dists):
                                if idx != r1 and idx != r2 and d > new_max:
                                    new_max = d
                            if new_max < current_max - 1e-12:
                                apply_cross(r1, pos1, r2, pos2)
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

    # Perturbation: move farthest-from-depot customer from max route to min route
    def perturb():
        nonlocal routes, route_dists
        current_max = max(route_dists)
        current_min = min(route_dists)
        max_idx = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12][0]
        min_idx = [i for i, d in enumerate(route_dists) if abs(d - current_min) < 1e-12][0]
        route_max = routes[max_idx]
        # Find farthest-from-depot customer in max route
        farthest_cust = max([c for c in route_max if c != 0], key=lambda c: distance_matrix[0][c])
        pos = route_max.index(farthest_cust)
        # Remove it
        route_max.pop(pos)
        route_dists[max_idx] = compute_route_dist(route_max)
        # Insert into min route at best position to minimize new max
        route_min = routes[min_idx]
        best_new_max = float('inf')
        best_pos = -1
        for insert_pos in range(1, len(route_min)):
            prev = route_min[insert_pos-1]
            succ = route_min[insert_pos]
            increase = distance_matrix[prev][farthest_cust] + distance_matrix[farthest_cust][succ] - distance_matrix[prev][succ]
            new_dist_min = route_dists[min_idx] + increase
            new_max = max(new_dist_min, route_dists[max_idx])
            if new_max < best_new_max:
                best_new_max = new_max
                best_pos = insert_pos
        if best_pos != -1:
            route_min.insert(best_pos, farthest_cust)
            route_dists[min_idx] = compute_route_dist(route_min)

    # Main loop
    max_restarts = min(10, n // 10)
    if max_restarts < 1:
        max_restarts = 1
    max_iter_per_restart = min(n * 10, 500)

    for restart in range(max_restarts):
        if restart > 0:
            # Reset to best solution and perturb
            routes = [list(r) for r in best_routes]
            route_dists = [compute_route_dist(r) for r in routes]
            perturb()
        # Local search with iteration limit
        for _ in range(max_iter_per_restart):
            old_max = max(route_dists)
            local_search()
            new_max = max(route_dists)
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            if new_max >= old_max - 1e-12:
                break  # no improvement
        if max(route_dists) < best_max - 1e-12:
            best_max = max(route_dists)
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    return best_routes