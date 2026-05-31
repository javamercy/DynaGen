import numpy as np
import random
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []

    def report_best_vrp(routes):
        pass

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(c, routes, route_dists):
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            other_max = 0.0
            for j, d in enumerate(route_dists):
                if j != r_idx and d > other_max:
                    other_max = d
            for pos in range(1, len(route)):
                pred = route[pos-1]
                succ = route[pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    # Regret construction
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))
    while unassigned:
        bests = []
        for c in unassigned:
            best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
            bests.append((-regret, c, best_route, best_pos, best_new_max))
        bests.sort(key=lambda x: (x[0], x[1]))
        _, c, best_route, best_pos, new_max = bests[0]
        route = routes[best_route]
        route.insert(best_pos, c)
        route_dists[best_route] = route_dist(route)
        unassigned.remove(c)
        report_best_vrp(routes)

    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)

    # Intra-route 2-opt
    for r_idx in range(truck_count):
        improved = True
        while improved:
            improved = False
            route = routes[r_idx]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new < old - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        route_dists[r_idx] = route_dist(route)
                        break
                if improved:
                    break
        cur_max = max(route_dists)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [route[:] for route in routes]
            report_best_vrp(routes)

    # Improvement loop (relocate and swap)
    max_iter_inner = n * truck_count
    for _ in range(max_iter_inner):
        improved_overall = False
        
        # Best-improvement relocate from longest route
        max_dist = max(route_dists)
        max_idx = route_dists.index(max_dist)
        best_move = None
        best_new_max = max_dist
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
                    if new_overall < best_new_max - 1e-12:
                        best_new_max = new_overall
                        best_move = (i, other_idx, pos, new_max_dist, new_other)
        if best_move is not None:
            i, other_idx, pos, new_max_dist, new_other = best_move
            c = route_max.pop(i)
            routes[other_idx].insert(pos, c)
            route_dists[max_idx] = new_max_dist
            route_dists[other_idx] = new_other
            for r_idx in [max_idx, other_idx]:
                improved = True
                while improved:
                    improved = False
                    route = routes[r_idx]
                    for a in range(1, len(route)-2):
                        for b in range(a+1, len(route)-1):
                            old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                            new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                            if new < old - 1e-12:
                                route[a:b+1] = reversed(route[a:b+1])
                                improved = True
                                route_dists[r_idx] = route_dist(route)
                                break
                        if improved:
                            break
            cur_max = max(route_dists)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [route[:] for route in routes]
                report_best_vrp(routes)
            improved_overall = True

        # Best-improvement swap
        if not improved_overall:
            best_swap = None
            best_new_max = max_dist
            for i in range(truck_count):
                for j in range(i+1, truck_count):
                    for pos_i in range(1, len(routes[i])-1):
                        c1 = routes[i][pos_i]
                        for pos_j in range(1, len(routes[j])-1):
                            c2 = routes[j][pos_j]
                            old1 = route_dists[i]
                            old2 = route_dists[j]
                            pred1 = routes[i][pos_i-1]
                            succ1 = routes[i][pos_i+1]
                            new_dist_i = old1 - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                            pred2 = routes[j][pos_j-1]
                            succ2 = routes[j][pos_j+1]
                            new_dist_j = old2 - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
                            other_max = 0.0
                            for k, d in enumerate(route_dists):
                                if k != i and k != j and d > other_max:
                                    other_max = d
                            new_overall = max(other_max, new_dist_i, new_dist_j)
                            if new_overall < best_new_max - 1e-12:
                                best_new_max = new_overall
                                best_swap = (i, pos_i, j, pos_j, new_dist_i, new_dist_j)
            if best_swap is not None:
                i, pos_i, j, pos_j, new_dist_i, new_dist_j = best_swap
                routes[i][pos_i], routes[j][pos_j] = routes[j][pos_j], routes[i][pos_i]
                route_dists[i] = new_dist_i
                route_dists[j] = new_dist_j
                for r_idx in [i, j]:
                    improved = True
                    while improved:
                        improved = False
                        route = routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved = True
                                    route_dists[r_idx] = route_dist(route)
                                    break
                            if improved:
                                break
                cur_max = max(route_dists)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [route[:] for route in routes]
                    report_best_vrp(routes)
                improved_overall = True

        if not improved_overall:
            break

    # Iterated local search
    outer_iterations = min(10, max(1, n // 10))
    for _ in range(outer_iterations):
        routes = [route[:] for route in best_routes]
        route_dists = [route_dist(r) for r in routes]
        # Perturb: random relocate from longest to another
        max_dist = max(route_dists)
        max_idx = route_dists.index(max_dist)
        route_max = routes[max_idx]
        if len(route_max) > 2:
            i = random.randint(1, len(route_max)-2)
            c = route_max.pop(i)
            candidates = [r for r in range(truck_count) if r != max_idx]
            if candidates:
                other_idx = random.choice(candidates)
                other_route = routes[other_idx]
                pos = random.randint(1, len(other_route)-1)
                other_route.insert(pos, c)
                route_dists[max_idx] = route_dist(route_max)
                route_dists[other_idx] = route_dist(other_route)
        # Re-optimize: 2-opt all routes
        for r_idx in range(truck_count):
            improved = True
            while improved:
                improved = False
                route = routes[r_idx]
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                        new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                        if new < old - 1e-12:
                            route[a:b+1] = reversed(route[a:b+1])
                            improved = True
                            route_dists[r_idx] = route_dist(route)
                            break
                    if improved:
                        break
        # Improvement loop again
        for _ in range(max_iter_inner):
            improved_overall = False
            # relocate
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_move = None
            best_new_max = max_dist
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
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_move = (i, other_idx, pos, new_max_dist, new_other)
            if best_move is not None:
                i, other_idx, pos, new_max_dist, new_other = best_move
                c = route_max.pop(i)
                routes[other_idx].insert(pos, c)
                route_dists[max_idx] = new_max_dist
                route_dists[other_idx] = new_other
                for r_idx in [max_idx, other_idx]:
                    improved = True
                    while improved:
                        improved = False
                        route = routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved = True
                                    route_dists[r_idx] = route_dist(route)
                                    break
                            if improved:
                                break
                cur_max = max(route_dists)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [route[:] for route in routes]
                    report_best_vrp(routes)
                improved_overall = True

            # swap
            if not improved_overall:
                best_swap = None
                best_new_max = max_dist
                for i in range(truck_count):
                    for j in range(i+1, truck_count):
                        for pos_i in range(1, len(routes[i])-1):
                            c1 = routes[i][pos_i]
                            for pos_j in range(1, len(routes[j])-1):
                                c2 = routes[j][pos_j]
                                old1 = route_dists[i]
                                old2 = route_dists[j]
                                pred1 = routes[i][pos_i-1]
                                succ1 = routes[i][pos_i+1]
                                new_dist_i = old1 - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                                pred2 = routes[j][pos_j-1]
                                succ2 = routes[j][pos_j+1]
                                new_dist_j = old2 - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
                                other_max = 0.0
                                for k, d in enumerate(route_dists):
                                    if k != i and k != j and d > other_max:
                                        other_max = d
                                new_overall = max(other_max, new_dist_i, new_dist_j)
                                if new_overall < best_new_max - 1e-12:
                                    best_new_max = new_overall
                                    best_swap = (i, pos_i, j, pos_j, new_dist_i, new_dist_j)
                if best_swap is not None:
                    i, pos_i, j, pos_j, new_dist_i, new_dist_j = best_swap
                    routes[i][pos_i], routes[j][pos_j] = routes[j][pos_j], routes[i][pos_i]
                    route_dists[i] = new_dist_i
                    route_dists[j] = new_dist_j
                    for r_idx in [i, j]:
                        improved = True
                        while improved:
                            improved = False
                            route = routes[r_idx]
                            for a in range(1, len(route)-2):
                                for b in range(a+1, len(route)-1):
                                    old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                    new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                    if new < old - 1e-12:
                                        route[a:b+1] = reversed(route[a:b+1])
                                        improved = True
                                        route_dists[r_idx] = route_dist(route)
                                        break
                                if improved:
                                    break
                    cur_max = max(route_dists)
                    if cur_max < best_max - 1e-12:
                        best_max = cur_max
                        best_routes = [route[:] for route in routes]
                        report_best_vrp(routes)
                    improved_overall = True

            if not improved_overall:
                break

    return best_routes