import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def total_dist(routes):
        return sum(route_dist(r) for r in routes)

    def construct_solution():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = list(range(1, n))
        random.shuffle(unassigned)
        for c in unassigned:
            best_cost = float('inf')
            best_route = -1
            best_pos = -1
            best_new_dist = 0.0
            for r_idx in range(truck_count):
                for pos in range(1, len(routes[r_idx])):
                    pred = routes[r_idx][pos-1]
                    succ = routes[r_idx][pos]
                    new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                    new_max = max(route_dists[r_idx], new_dist)
                    if new_max < best_cost - 1e-12 or (abs(new_max - best_cost) < 1e-12 and new_dist < best_new_dist - 1e-12):
                        best_cost = new_max
                        best_route = r_idx
                        best_pos = pos
                        best_new_dist = new_dist
            routes[best_route].insert(best_pos, c)
            route_dists[best_route] = route_dist(routes[best_route])
        return routes, route_dists

    def improve(routes, route_dists):
        best_routes = [route[:] for route in routes]
        best_max = max(route_dists)
        best_total = total_dist(routes)
        improved = True
        while improved:
            improved = False
            # relocate best-improvement on max route
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_move = None
            best_new_max = max_dist
            best_new_total = best_total
            route_max = routes[max_idx]
            for i in range(1, len(route_max)-1):
                c = route_max[i]
                pred = route_max[i-1]
                succ = route_max[i+1]
                new_max_dist = route_dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for pos in range(1, len(other_route)):
                        pred_o = other_route[pos-1]
                        succ_o = other_route[pos]
                        new_other = route_dists[other_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                        other_max = 0.0
                        for j, d in enumerate(route_dists):
                            if j != max_idx and j != other_idx and d > other_max:
                                other_max = d
                        new_overall = max(other_max, new_max_dist, new_other)
                        if new_overall < best_new_max - 1e-12 or (abs(new_overall - best_new_max) < 1e-12 and new_other + new_max_dist < best_new_total - 1e-12):
                            best_new_max = new_overall
                            best_new_total = new_other + new_max_dist + sum(route_dists[j] for j in range(truck_count) if j != max_idx and j != other_idx)
                            best_move = (i, other_idx, pos, new_max_dist, new_other)
            if best_move is not None:
                i, other_idx, pos, new_max_dist, new_other = best_move
                c = route_max.pop(i)
                routes[other_idx].insert(pos, c)
                route_dists[max_idx] = new_max_dist
                route_dists[other_idx] = new_other
                improved = True
            if not improved:
                # swap best-improvement on max route
                max_dist = max(route_dists)
                max_idx = route_dists.index(max_dist)
                best_swap = None
                best_new_max = max_dist
                best_new_total = best_total
                route_max = routes[max_idx]
                for i in range(1, len(route_max)-1):
                    c1 = route_max[i]
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = routes[other_idx]
                        for j in range(1, len(other_route)-1):
                            c2 = other_route[j]
                            old1 = route_dists[max_idx]
                            old2 = route_dists[other_idx]
                            pred1 = route_max[i-1]
                            succ1 = route_max[i+1]
                            new_dist_max = old1 - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                            pred2 = other_route[j-1]
                            succ2 = other_route[j+1]
                            new_dist_other = old2 - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
                            other_max = 0.0
                            for k, d in enumerate(route_dists):
                                if k != max_idx and k != other_idx and d > other_max:
                                    other_max = d
                            new_overall = max(other_max, new_dist_max, new_dist_other)
                            if new_overall < best_new_max - 1e-12 or (abs(new_overall - best_new_max) < 1e-12 and new_dist_max + new_dist_other < best_new_total - 1e-12):
                                best_new_max = new_overall
                                best_new_total = new_dist_max + new_dist_other + sum(route_dists[k] for k in range(truck_count) if k != max_idx and k != other_idx)
                                best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
                if best_swap is not None:
                    i, other_idx, j, new_dist_max, new_dist_other = best_swap
                    route_max = routes[max_idx]
                    other_route = routes[other_idx]
                    route_max[i], other_route[j] = other_route[j], route_max[i]
                    route_dists[max_idx] = new_dist_max
                    route_dists[other_idx] = new_dist_other
                    improved = True
            if not improved:
                # 2-opt* best-improvement on max route
                max_dist = max(route_dists)
                max_idx = route_dists.index(max_dist)
                best_cross = None
                best_new_max = max_dist
                best_new_total = best_total
                route_max = routes[max_idx]
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for i in range(1, len(route_max)-1):
                        for j in range(1, len(other_route)-1):
                            if route_max[-1] != 0 or other_route[-1] != 0:
                                continue
                            old1 = distance_matrix[route_max[i], route_max[i+1]]
                            old2 = distance_matrix[other_route[j], other_route[j+1]]
                            new1 = distance_matrix[route_max[i], other_route[j+1]]
                            new2 = distance_matrix[other_route[j], route_max[i+1]]
                            new_dist_max = route_dists[max_idx] - old1 + new1
                            new_dist_other = route_dists[other_idx] - old2 + new2
                            other_max = 0.0
                            for k, d in enumerate(route_dists):
                                if k != max_idx and k != other_idx and d > other_max:
                                    other_max = d
                            new_overall = max(other_max, new_dist_max, new_dist_other)
                            if new_overall < best_new_max - 1e-12 or (abs(new_overall - best_new_max) < 1e-12 and new_dist_max + new_dist_other < best_new_total - 1e-12):
                                best_new_max = new_overall
                                best_new_total = new_dist_max + new_dist_other + sum(route_dists[k] for k in range(truck_count) if k != max_idx and k != other_idx)
                                best_cross = (i, other_idx, j, new_dist_max, new_dist_other)
                if best_cross is not None:
                    i, other_idx, j, new_dist_max, new_dist_other = best_cross
                    route_max = routes[max_idx]
                    other_route = routes[other_idx]
                    new_route_max = route_max[:i+1] + other_route[j+1:]
                    new_route_other = other_route[:j+1] + route_max[i+1:]
                    routes[max_idx] = new_route_max
                    routes[other_idx] = new_route_other
                    route_dists[max_idx] = route_dist(new_route_max)
                    route_dists[other_idx] = route_dist(new_route_other)
                    improved = True
            if improved:
                for r_idx in [max_idx, other_idx]:
                    # 2-opt on single route
                    route = routes[r_idx]
                    improved_local = True
                    while improved_local:
                        improved_local = False
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved_local = True
                                    route_dists[r_idx] = route_dist(route)
                                    break
                            if improved_local:
                                break
                cur_max = max(route_dists)
                cur_total = total_dist(routes)
                if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [route[:] for route in routes]
                    report_best_vrp(best_routes)
            else:
                break
        return best_routes, [route_dist(r) for r in best_routes]

    # Initial solution
    routes, route_dists = construct_solution()
    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)
    best_total = total_dist(routes)
    report_best_vrp(best_routes)
    routes, route_dists = improve(routes, route_dists)
    cur_max = max(route_dists)
    cur_total = total_dist(routes)
    if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
        best_max = cur_max
        best_total = cur_total
        best_routes = [route[:] for route in routes]
        report_best_vrp(best_routes)

    # Outer loop with perturbation
    outer_iter = min(20, n * 2)
    removal_size = max(1, int((n-1) * 0.15))
    for _ in range(outer_iter):
        routes = [route[:] for route in best_routes]
        route_dists = [route_dist(r) for r in routes]
        # Random removal
        unassigned = list(range(1, n))
        random.shuffle(unassigned)
        to_remove = unassigned[:removal_size]
        for c in to_remove:
            for r_idx in range(truck_count):
                if c in routes[r_idx]:
                    pos = routes[r_idx].index(c)
                    pred = routes[r_idx][pos-1]
                    succ = routes[r_idx][pos+1]
                    route_dists[r_idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                    routes[r_idx].pop(pos)
                    break
        # Greedy reinsertion
        while to_remove:
            best_cost = float('inf')
            best_c = -1
            best_route = -1
            best_pos = -1
            best_new_dist = 0.0
            for c in to_remove:
                for r_idx in range(truck_count):
                    for pos in range(1, len(routes[r_idx])):
                        pred = routes[r_idx][pos-1]
                        succ = routes[r_idx][pos]
                        new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                        new_max = max(route_dists[r_idx], new_dist)
                        if new_max < best_cost - 1e-12 or (abs(new_max - best_cost) < 1e-12 and new_dist < best_new_dist - 1e-12):
                            best_cost = new_max
                            best_c = c
                            best_route = r_idx
                            best_pos = pos
                            best_new_dist = new_dist
            routes[best_route].insert(best_pos, best_c)
            route_dists[best_route] = route_dist(routes[best_route])
            to_remove.remove(best_c)
        routes, route_dists = improve(routes, route_dists)
        cur_max = max(route_dists)
        cur_total = total_dist(routes)
        if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
            best_max = cur_max
            best_total = cur_total
            best_routes = [route[:] for route in routes]
            report_best_vrp(best_routes)
    return best_routes