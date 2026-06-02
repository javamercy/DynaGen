import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.tolist()
    def route_distance(route):
        if len(route) < 2:
            return 0
        return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))
    def objective(routes):
        return max(route_distance(r) for r in routes)
    # Construction
    routes = [[0,0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    while unassigned:
        best = None
        for node in unassigned:
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [node] + route[pos:]
                    new_dist = route_distance(new_route)
                    new_max = new_dist
                    for rr in range(truck_count):
                        if rr != r_idx:
                            d = route_distance(routes[rr])
                            if d > new_max:
                                new_max = d
                    new_total = sum(route_distance(routes[rr]) for rr in range(truck_count) if rr != r_idx) + new_dist
                    candidate = (new_max, new_total, r_idx, pos, node)
                    if best is None or candidate < best:
                        best = candidate
        new_max, new_total, r_idx, pos, node = best
        routes[r_idx].insert(pos, node)
        unassigned.remove(node)
    best_routes = [list(r) for r in routes]
    best_obj = objective(best_routes)
    report_best_vrp(best_routes)
    # Improvement
    max_iter = 3
    for iteration in range(max_iter):
        current_routes = [list(r) for r in best_routes]
        route_dists = [route_distance(r) for r in current_routes]
        current_max = max(route_dists)
        current_total = sum(route_dists)
        improved_this_iter = False
        # Relocate: best improvement
        best_delta = 0
        best_move = None
        best_new_max = None
        best_new_total = None
        for i in range(truck_count):
            if len(current_routes[i]) <= 2:
                continue
            for ci_idx in range(1, len(current_routes[i])-1):
                ci = current_routes[i][ci_idx]
                for j in range(truck_count):
                    if i == j:
                        continue
                    for cj_idx in range(1, len(current_routes[j])):
                        new_route_i = current_routes[i][:ci_idx] + current_routes[i][ci_idx+1:]
                        new_route_j = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx:]
                        new_dist_i = route_distance(new_route_i)
                        new_dist_j = route_distance(new_route_j)
                        new_max = new_dist_i
                        if new_dist_j > new_max:
                            new_max = new_dist_j
                        for k in range(truck_count):
                            if k == i or k == j:
                                continue
                            d = route_dists[k]
                            if d > new_max:
                                new_max = d
                        delta = current_max - new_max
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (i, ci_idx, j, cj_idx)
                            best_new_max = new_max
                            best_new_total = current_total - route_dists[i] - route_dists[j] + new_dist_i + new_dist_j
                        elif delta == best_delta and best_move is not None:
                            if new_max < best_new_max or (new_max == best_new_max and (current_total - route_dists[i] - route_dists[j] + new_dist_i + new_dist_j) < best_new_total):
                                best_delta = delta
                                best_move = (i, ci_idx, j, cj_idx)
                                best_new_max = new_max
                                best_new_total = current_total - route_dists[i] - route_dists[j] + new_dist_i + new_dist_j
        if best_delta > 0 and best_move is not None:
            i, ci_idx, j, cj_idx = best_move
            ci = current_routes[i][ci_idx]
            current_routes[i] = current_routes[i][:ci_idx] + current_routes[i][ci_idx+1:]
            current_routes[j] = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx:]
            route_dists[i] = route_distance(current_routes[i])
            route_dists[j] = route_distance(current_routes[j])
            current_max = max(route_dists)
            current_total = sum(route_dists)
            improved_this_iter = True
            if current_max < best_obj:
                best_obj = current_max
                best_routes = [list(r) for r in current_routes]
                report_best_vrp(best_routes)
        # Swap: best improvement
        best_delta = 0
        best_move = None
        best_new_max = None
        best_new_total = None
        for i in range(truck_count):
            if len(current_routes[i]) <= 2:
                continue
            for ci_idx in range(1, len(current_routes[i])-1):
                ci = current_routes[i][ci_idx]
                for j in range(i+1, truck_count):
                    if len(current_routes[j]) <= 2:
                        continue
                    for cj_idx in range(1, len(current_routes[j])-1):
                        cj = current_routes[j][cj_idx]
                        new_route_i = current_routes[i][:ci_idx] + [cj] + current_routes[i][ci_idx+1:]
                        new_route_j = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx+1:]
                        new_dist_i = route_distance(new_route_i)
                        new_dist_j = route_distance(new_route_j)
                        new_max = new_dist_i
                        if new_dist_j > new_max:
                            new_max = new_dist_j
                        for k in range(truck_count):
                            if k == i or k == j:
                                continue
                            d = route_dists[k]
                            if d > new_max:
                                new_max = d
                        delta = current_max - new_max
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (i, ci_idx, j, cj_idx)
                            best_new_max = new_max
                            best_new_total = current_total - route_dists[i] - route_dists[j] + new_dist_i + new_dist_j
                        elif delta == best_delta and best_move is not None:
                            if new_max < best_new_max or (new_max == best_new_max and (current_total - route_dists[i] - route_dists[j] + new_dist_i + new_dist_j) < best_new_total):
                                best_delta = delta
                                best_move = (i, ci_idx, j, cj_idx)
                                best_new_max = new_max
                                best_new_total = current_total - route_dists[i] - route_dists[j] + new_dist_i + new_dist_j
        if best_delta > 0 and best_move is not None:
            i, ci_idx, j, cj_idx = best_move
            ci = current_routes[i][ci_idx]
            cj = current_routes[j][cj_idx]
            current_routes[i] = current_routes[i][:ci_idx] + [cj] + current_routes[i][ci_idx+1:]
            current_routes[j] = current_routes[j][:cj_idx] + [ci] + current_routes[j][cj_idx+1:]
            route_dists[i] = route_distance(current_routes[i])
            route_dists[j] = route_distance(current_routes[j])
            current_max = max(route_dists)
            current_total = sum(route_dists)
            improved_this_iter = True
            if current_max < best_obj:
                best_obj = current_max
                best_routes = [list(r) for r in current_routes]
                report_best_vrp(best_routes)
        # Intra-route 2-opt for each route (best improvement per route)
        for r_idx in range(truck_count):
            route = current_routes[r_idx]
            if len(route) <= 3:
                continue
            best_gain = 0
            best_ij = None
            old_dist = route_dists[r_idx]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    gain = old_dist - new_dist
                    if gain > best_gain:
                        best_gain = gain
                        best_ij = (i, j)
            if best_gain > 0:
                i, j = best_ij
                route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                current_routes[r_idx] = route
                route_dists[r_idx] = route_distance(route)
                current_max = max(route_dists)
                current_total = sum(route_dists)
                improved_this_iter = True
                if current_max < best_obj:
                    best_obj = current_max
                    best_routes = [list(r) for r in current_routes]
                    report_best_vrp(best_routes)
        if not improved_this_iter:
            break
    return best_routes