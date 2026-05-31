import numpy as np
import random
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []

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

    def two_opt(route):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new < old - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        break
                if improved:
                    break
        return route

    def improve(routes, route_dists):
        max_iter = n * truck_count
        for _ in range(max_iter):
            improved = False
            # best-improvement relocate from longest route
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
                routes[max_idx] = two_opt(routes[max_idx])
                routes[other_idx] = two_opt(routes[other_idx])
                route_dists[max_idx] = route_dist(routes[max_idx])
                route_dists[other_idx] = route_dist(routes[other_idx])
                improved = True
                continue
            # best-improvement swap
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_swap = None
            best_new_max = max_dist
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
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
            if best_swap is not None:
                i, other_idx, j, new_dist_max, new_dist_other = best_swap
                route_max = routes[max_idx]
                other_route = routes[other_idx]
                c1 = route_max[i]
                c2 = other_route[j]
                route_max[i] = c2
                other_route[j] = c1
                route_dists[max_idx] = new_dist_max
                route_dists[other_idx] = new_dist_other
                routes[max_idx] = two_opt(routes[max_idx])
                routes[other_idx] = two_opt(routes[other_idx])
                route_dists[max_idx] = route_dist(routes[max_idx])
                route_dists[other_idx] = route_dist(routes[other_idx])
                improved = True
                continue
            # best-improvement 2-opt*
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_cross = None
            best_new_max = max_dist
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
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
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
                routes[max_idx] = two_opt(routes[max_idx])
                routes[other_idx] = two_opt(routes[other_idx])
                route_dists[max_idx] = route_dist(routes[max_idx])
                route_dists[other_idx] = route_dist(routes[other_idx])
                improved = True
                continue
            if not improved:
                break
        return routes, route_dists

    # Construction
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

    # Initial 2-opt on all routes
    for r_idx in range(truck_count):
        routes[r_idx] = two_opt(routes[r_idx])
        route_dists[r_idx] = route_dist(routes[r_idx])

    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)

    # Main improvement
    routes, route_dists = improve(routes, route_dists)
    cur_max = max(route_dists)
    if cur_max < best_max - 1e-12:
        best_max = cur_max
        best_routes = [route[:] for route in routes]

    # Iterated local search
    outer_iterations = min(10, max(1, n // 10))
    for _ in range(outer_iterations):
        routes = [route[:] for route in best_routes]
        route_dists = [route_dist(r) for r in routes]
        # Perturb
        max_dist = max(route_dists)
        max_idx = route_dists.index(max_dist)
        route_max = routes[max_idx]
        if len(route_max) > 2:
            i = random.randint(1, len(route_max)-2)
            c = route_max.pop(i)
            candidates = [r for r in range(truck_count) if r != max_idx]
            other_idx = random.choice(candidates)
            other_route = routes[other_idx]
            pos = random.randint(1, len(other_route)-1)
            other_route.insert(pos, c)
            route_dists[max_idx] = route_dist(route_max)
            route_dists[other_idx] = route_dist(other_route)
        # Re-apply 2-opt
        for r_idx in range(truck_count):
            routes[r_idx] = two_opt(routes[r_idx])
            route_dists[r_idx] = route_dist(routes[r_idx])
        # Improve again
        routes, route_dists = improve(routes, route_dists)
        cur_max = max(route_dists)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [route[:] for route in routes]

    return best_routes