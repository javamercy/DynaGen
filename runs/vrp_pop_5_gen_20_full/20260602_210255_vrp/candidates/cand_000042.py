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
                new_max = new_route_dist
                for other_idx, d in enumerate(route_dists):
                    if other_idx != r_idx and d > new_max:
                        new_max = d
                if new_max < best_new_max or (abs(new_max - best_new_max) < 1e-12 and r_idx < best_route_idx):
                    best_new_max = new_max
                    best_route_idx = r_idx
                    best_pos = pos
        routes[best_route_idx].insert(best_pos, cust)
        route_dists[best_route_idx] = compute_route_dist(routes[best_route_idx])

    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)

    # move helpers
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

    # perturbation: relocate k customers from longest route to shortest routes
    def pert_relocate_k(routes, route_dists, k=3):
        # find longest route
        current_max = max(route_dists)
        max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
        r_idx = max_routes[0]
        route = routes[r_idx]
        customers = [node for node in route if node != 0]
        if not customers:
            return
        # pick up to k customers to move (the ones with highest indices to break ties deterministically)
        customers.sort(reverse=True)
        custs_to_move = customers[:min(k, len(customers))]
        for cust in custs_to_move:
            pos = route.index(cust)
            route.pop(pos)
        route_dists[r_idx] = compute_route_dist(route)
        # insert each into shortest routes
        for cust in custs_to_move:
            # find route with smallest distance
            min_route_idx = min(range(truck_count), key=lambda i: (route_dists[i], i))
            best_increase = float('inf')
            best_pos = -1
            for pos in range(1, len(routes[min_route_idx])):
                prev = routes[min_route_idx][pos-1]
                succ = routes[min_route_idx][pos]
                increase = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
                if increase < best_increase:
                    best_increase = increase
                    best_pos = pos
            routes[min_route_idx].insert(best_pos, cust)
            route_dists[min_route_idx] = compute_route_dist(routes[min_route_idx])

    # simulated annealing local search
    def local_search_sa(routes, route_dists, temp_start, temp_end, max_iter):
        T = temp_start
        cooling = (temp_end / temp_start) ** (1.0 / max_iter) if max_iter > 0 else 1.0
        current_max = max(route_dists)
        current_dists = list(route_dists)
        for it in range(max_iter):
            if T < 1e-12:
                T = 1e-12
            # first-improvement over moves
            improved = False
            # try relocate
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)-1):
                    cust = route[pos]
                    prev = route[pos-1]
                    succ = route[pos+1]
                    removal_change = distance_matrix[prev][succ] - (distance_matrix[prev][cust] + distance_matrix[cust][succ])
                    new_dist_r = current_dists[r_idx] + removal_change
                    for other_idx in range(truck_count):
                        if other_idx == r_idx:
                            continue
                        other_route = routes[other_idx]
                        for insert_pos in range(1, len(other_route)):
                            prev2 = other_route[insert_pos-1]
                            succ2 = other_route[insert_pos]
                            insertion_change = distance_matrix[prev2][cust] + distance_matrix[cust][succ2] - distance_matrix[prev2][succ2]
                            new_dist_other = current_dists[other_idx] + insertion_change
                            new_max = max(new_dist_r, new_dist_other)
                            for idx, d in enumerate(current_dists):
                                if idx != r_idx and idx != other_idx and d > new_max:
                                    new_max = d
                            delta = new_max - current_max
                            if delta < 0 or random.random() < np.exp(-delta / T):
                                apply_relocate(routes, r_idx, pos, cust, other_idx, insert_pos)
                                current_dists[r_idx] = route_dists[r_idx]
                                current_dists[other_idx] = route_dists[other_idx]
                                current_max = max(current_dists)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                # try swap
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
                                new_dist_r1 = current_dists[r1] + remove1_change + insert1_change
                                new_dist_r2 = current_dists[r2] + remove2_change + insert2_change
                                new_max = max(new_dist_r1, new_dist_r2)
                                for idx, d in enumerate(current_dists):
                                    if idx != r1 and idx != r2 and d > new_max:
                                        new_max = d
                                delta = new_max - current_max
                                if delta < 0 or random.random() < np.exp(-delta / T):
                                    apply_swap(routes, r1, pos1, r2, pos2)
                                    current_dists[r1] = route_dists[r1]
                                    current_dists[r2] = route_dists[r2]
                                    current_max = max(current_dists)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                # try 2-opt on each route
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            old_edges = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                            new_edges = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                            change = new_edges - old_edges
                            new_dist = current_dists[r_idx] + change
                            new_max = max(new_dist, max(d for idx, d in enumerate(current_dists) if idx != r_idx))
                            delta = new_max - current_max
                            if delta < 0 or random.random() < np.exp(-delta / T):
                                apply_two_opt(routes, r_idx, i, j)
                                current_dists[r_idx] = route_dists[r_idx]
                                current_max = max(current_dists)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                # try cross-exchange between two routes
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
                                new_dist_r1 = current_dists[r1] + delta1
                                new_dist_r2 = current_dists[r2] + delta2
                                new_max = max(new_dist_r1, new_dist_r2)
                                for idx, d in enumerate(current_dists):
                                    if idx != r1 and idx != r2 and d > new_max:
                                        new_max = d
                                delta = new_max - current_max
                                if delta < 0 or random.random() < np.exp(-delta / T):
                                    apply_cross(routes, r1, pos1, r2, pos2)
                                    current_dists[r1] = route_dists[r1]
                                    current_dists[r2] = route_dists[r2]
                                    current_max = max(current_dists)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
            T *= cooling
        return current_dists, current_max

    # main loop with restarts
    max_restarts = 3
    for restart in range(max_restarts + 1):
        if restart == 0:
            cur_routes = [list(r) for r in best_routes]
            cur_dists = [compute_route_dist(r) for r in cur_routes]
        else:
            cur_routes = [list(r) for r in best_routes]
            cur_dists = [compute_route_dist(r) for r in cur_routes]
            pert_relocate_k(cur_routes, cur_dists, k=3)

        temp_start = best_max * 0.1
        temp_end = 0.001
        max_iter = min(n * 20, 1000)
        cur_dists, cur_max = local_search_sa(cur_routes, cur_dists, temp_start, temp_end, max_iter)

        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [list(r) for r in cur_routes]
            report_best_vrp(best_routes)

    return best_routes