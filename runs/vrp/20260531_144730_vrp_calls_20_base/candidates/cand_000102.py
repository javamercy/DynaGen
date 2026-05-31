import numpy as np
import random
import math

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

    # Initial construction: regret heuristic
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

    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)

    # First-improvement local search
    def improve(routes, route_dists):
        nonlocal best_routes, best_max
        # Intra-route 2-opt on all routes (limited passes)
        for r_idx in range(truck_count):
            for _ in range(10):  # limit passes
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
                if not improved:
                    break
        # Inter-route improvement: first-improvement relocate and swap
        max_iter = n * truck_count
        for _ in range(max_iter):
            improved_overall = False
            # First-improvement relocate from longest route
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            route_max = routes[max_idx]
            found = False
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
                        if new_overall < best_max - 1e-12:
                            # Accept first improvement
                            c = route_max.pop(i)
                            other_route.insert(pos, c)
                            route_dists[max_idx] = new_max_dist
                            route_dists[other_idx] = new_other
                            # Intra-route 2-opt on affected routes
                            for r_idx in [max_idx, other_idx]:
                                for _ in range(10):
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
                                    if not improved:
                                        break
                            cur_max = max(route_dists)
                            if cur_max < best_max - 1e-12:
                                best_max = cur_max
                                best_routes = [route[:] for route in routes]
                                report_best_vrp(best_routes)
                            found = True
                            break
                    if found:
                        break
                if found:
                    improved_overall = True
                    break
            if not improved_overall:
                # First-improvement swap from longest route
                max_dist = max(route_dists)
                max_idx = route_dists.index(max_dist)
                route_max = routes[max_idx]
                found = False
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
                            if new_overall < best_max - 1e-12:
                                route_max[i] = c2
                                other_route[j] = c1
                                route_dists[max_idx] = new_dist_max
                                route_dists[other_idx] = new_dist_other
                                for r_idx in [max_idx, other_idx]:
                                    for _ in range(10):
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
                                        if not improved:
                                            break
                                cur_max = max(route_dists)
                                if cur_max < best_max - 1e-12:
                                    best_max = cur_max
                                    best_routes = [route[:] for route in routes]
                                    report_best_vrp(best_routes)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        improved_overall = True
                        break
            if not improved_overall:
                break
        return routes, route_dists

    # Initial improvement
    routes, route_dists = improve(routes, route_dists)

    # Outer loop with adaptive perturbation and simulated annealing
    outer_iter = min(20, n)
    stagnation_counter = 0
    num_removals_base = max(1, int((n-1) * 0.15))
    max_removals = max(1, int((n-1) * 0.4))
    temperature = best_max * 0.2
    cooling = 0.95
    for it in range(outer_iter):
        routes = [route[:] for route in best_routes]
        route_dists = [route_dist(r) for r in routes]
        extra = stagnation_counter * int((n-1) * 0.05)
        num_remove = min(max_removals, num_removals_base + extra)
        # Worst removal
        removal_costs = []
        for c in range(1, n):
            for r_idx, route in enumerate(routes):
                if c in route:
                    pos = route.index(c)
                    pred = route[pos-1]
                    succ = route[pos+1]
                    cost = distance_matrix[pred, c] + distance_matrix[c, succ] - distance_matrix[pred, succ]
                    removal_costs.append((cost, c, r_idx, pos))
                    break
        removal_costs.sort(key=lambda x: -x[0])
        num_worst = max(1, len(removal_costs)//2)
        candidates = removal_costs[:num_worst]
        random.shuffle(candidates)
        to_remove = [c[1] for c in candidates[:num_remove]]
        for c in to_remove:
            for r_idx in range(truck_count):
                if c in routes[r_idx]:
                    pos = routes[r_idx].index(c)
                    pred = routes[r_idx][pos-1]
                    succ = routes[r_idx][pos+1]
                    route_dists[r_idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                    routes[r_idx].pop(pos)
                    break
        unassigned = to_remove[:]
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
        new_routes, new_dists = improve(routes, route_dists)
        cur_max = max(new_dists)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [route[:] for route in new_routes]
            report_best_vrp(best_routes)
            stagnation_counter = 0
            temperature = best_max * 0.2
        else:
            delta = cur_max - best_max
            if delta > 0 and temperature > 1e-6:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    routes = new_routes
                    route_dists = new_dists
            stagnation_counter += 1
        temperature *= cooling

    return best_routes