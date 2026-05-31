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

    # Initial construction: cheapest insertion with random order
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))
    random.shuffle(unassigned)
    for c in unassigned:
        best_new_max, best_route, best_pos, _ = best_insertion(c, routes, route_dists)
        if best_route == -1:
            best_new_max = float('inf')
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    pred = route[pos-1]
                    succ = route[pos]
                    new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                    new_max = max(route_dists[r_idx], new_dist)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_route = r_idx
                        best_pos = pos
        route = routes[best_route]
        route.insert(best_pos, c)
        route_dists[best_route] = route_dist(route)

    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)
    best_total = total_dist(routes)
    report_best_vrp(best_routes)

    def improve(routes, route_dists):
        nonlocal best_routes, best_max, best_total
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
        # Inter-route improvement loop
        max_iter = n * truck_count
        for _ in range(max_iter):
            improved_overall = False
            # Best-improvement relocate
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_move = None
            best_new_max = max_dist
            best_new_total = total_dist(routes)
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
                            best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_max_dist + new_other
                            best_move = (i, other_idx, pos, new_max_dist, new_other)
                        elif abs(new_overall - best_new_max) < 1e-12:
                            new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_max_dist + new_other
                            if new_total < best_new_total - 1e-12:
                                best_new_max = new_overall
                                best_new_total = new_total
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
                cur_total = total_dist(routes)
                if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [route[:] for route in routes]
                    report_best_vrp(best_routes)
                improved_overall = True

            if not improved_overall:
                # Best-improvement swap
                max_dist = max(route_dists)
                max_idx = route_dists.index(max_dist)
                best_swap = None
                best_new_max = max_dist
                best_new_total = total_dist(routes)
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
                                best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                                best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
                            elif abs(new_overall - best_new_max) < 1e-12:
                                new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                                if new_total < best_new_total - 1e-12:
                                    best_new_max = new_overall
                                    best_new_total = new_total
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
                    cur_total = total_dist(routes)
                    if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                        best_max = cur_max
                        best_total = cur_total
                        best_routes = [route[:] for route in routes]
                        report_best_vrp(best_routes)
                    improved_overall = True

            if not improved_overall:
                # Best-improvement 2-opt*
                max_dist = max(route_dists)
                max_idx = route_dists.index(max_dist)
                best_cross = None
                best_new_max = max_dist
                best_new_total = total_dist(routes)
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
                                best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                                best_cross = (i, other_idx, j, new_dist_max, new_dist_other)
                            elif abs(new_overall - best_new_max) < 1e-12:
                                new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                                if new_total < best_new_total - 1e-12:
                                    best_new_max = new_overall
                                    best_new_total = new_total
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
                    cur_total = total_dist(routes)
                    if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                        best_max = cur_max
                        best_total = cur_total
                        best_routes = [route[:] for route in routes]
                        report_best_vrp(best_routes)
                    improved_overall = True

            if not improved_overall:
                break
        return routes, route_dists

    routes, route_dists = improve(routes, route_dists)

    # ILS phase with adaptive perturbation
    outer_iterations = min(50, n * 2)
    stagnation_limit = max(5, n // 20)
    max_perturbation = max(5, n // 10)
    last_improvement = 0
    perturbation_strength = 1
    for iteration in range(outer_iterations):
        routes = [route[:] for route in best_routes]
        route_dists = [route_dist(r) for r in routes]
        if iteration - last_improvement > stagnation_limit:
            perturbation_strength = min(perturbation_strength + 1, max_perturbation)
        for _ in range(perturbation_strength):
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            route_max = routes[max_idx]
            if len(route_max) <= 2:
                break
            i = random.randint(1, len(route_max)-2)
            c = route_max.pop(i)
            candidates = [r for r in range(truck_count) if r != max_idx]
            other_idx = random.choice(candidates)
            other_route = routes[other_idx]
            pos = random.randint(1, len(other_route)-1)
            other_route.insert(pos, c)
            route_dists[max_idx] = route_dist(route_max)
            route_dists[other_idx] = route_dist(other_route)
        routes, route_dists = improve(routes, route_dists)
        cur_max = max(route_dists)
        cur_total = total_dist(routes)
        if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
            best_max = cur_max
            best_total = cur_total
            best_routes = [route[:] for route in routes]
            report_best_vrp(best_routes)
            last_improvement = iteration
            perturbation_strength = 1
        if iteration - last_improvement > stagnation_limit * 3:
            break

    return best_routes