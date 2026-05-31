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

    def best_insertion(c, routes, route_dists, noise=False, noise_scale=0.0):
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
                if noise:
                    noise_val = random.uniform(-noise_scale, noise_scale)
                    new_max += noise_val
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    best_routes = None
    best_max = float('inf')
    restarts = max(1, min(5, n // 10))
    # Adaptive parameters
    pert_lower = 0.2
    pert_upper = 0.5
    noise_scale_init = 0.02 * np.mean(distance_matrix[distance_matrix > 0]) if np.count_nonzero(distance_matrix) > 1 else 1.0
    cooling_factor_normal = 0.95
    cooling_factor_low = 0.9
    reheat_threshold = 5
    no_improve_count = 0

    for restart in range(restarts):
        noise_scale = noise_scale_init * (1.0 - 0.005 * restart)
        noise_scale = max(noise_scale, 0.0)
        # Randomized regret construction with noise
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = list(range(1, n))
        while unassigned:
            bests = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists, noise=True, noise_scale=noise_scale)
                if best_route == -1:
                    continue
                regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                bests.append((-regret, c, best_route, best_pos, best_new_max))
            bests.sort(key=lambda x: (x[0], random.random()))
            _, c, best_route, best_pos, new_max = bests[0]
            route = routes[best_route]
            route.insert(best_pos, c)
            route_dists[best_route] = route_dist(route)
            unassigned.remove(c)

        # Improvement function (same as parent)
        def improve(routes, route_dists):
            nonlocal best_routes, best_max, no_improve_count, pert_lower, pert_upper
            # Intra-route 2-opt on all routes
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
                        report_best_vrp(best_routes)
                        no_improve_count = 0
                        pert_lower = 0.2
                        pert_upper = 0.5
                    improved_overall = True

                # Best-improvement swap
                if not improved_overall:
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
                            report_best_vrp(best_routes)
                            no_improve_count = 0
                            pert_lower = 0.2
                            pert_upper = 0.5
                        improved_overall = True

                # Best-improvement 2-opt* (swap suffixes)
                if not improved_overall:
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
                            report_best_vrp(best_routes)
                            no_improve_count = 0
                            pert_lower = 0.2
                            pert_upper = 0.5
                        improved_overall = True

                if not improved_overall:
                    break
            return routes, route_dists

        # Initial improvement
        routes, route_dists = improve(routes, route_dists)
        cur_max = max(route_dists)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [route[:] for route in routes]
            report_best_vrp(best_routes)
            no_improve_count = 0
            pert_lower = 0.2
            pert_upper = 0.5

        # Outer loop with perturbations and simulated annealing acceptance
        outer_iterations = max(5, min(20, n // 5))
        temperature = best_max * 0.2
        cooling_factor = cooling_factor_normal
        for it in range(outer_iterations):
            # Start from current best solution (to encourage exploration, sometimes start from best)
            if random.random() < 0.7:
                routes = [route[:] for route in best_routes]
                route_dists = [route_dist(r) for r in routes]
            else:
                routes = [route[:] for route in routes]
            # Dynamic perturbation size
            num_remove = random.randint(int(pert_lower * (n-1)), int(pert_upper * (n-1)))
            num_remove = max(1, min(num_remove, n-1))
            customers = list(range(1, n))
            random.shuffle(customers)
            to_remove = customers[:num_remove]
            temp_routes = [route[:] for route in routes]
            temp_dists = route_dists[:]
            for c in to_remove:
                for r_idx in range(truck_count):
                    if c in temp_routes[r_idx]:
                        pos = temp_routes[r_idx].index(c)
                        pred = temp_routes[r_idx][pos-1]
                        succ = temp_routes[r_idx][pos+1]
                        temp_dists[r_idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                        temp_routes[r_idx].pop(pos)
                        break
            # Repair using regret (no noise for deterministic improvement)
            unassigned = to_remove[:]
            while unassigned:
                bests = []
                for c in unassigned:
                    best_new_max, best_route, best_pos, second_new_max = best_insertion(c, temp_routes, temp_dists, noise=False)
                    if best_route == -1:
                        continue
                    regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                    bests.append((-regret, c, best_route, best_pos, best_new_max))
                bests.sort(key=lambda x: (x[0], random.random()))
                _, c, best_route, best_pos, new_max = bests[0]
                route = temp_routes[best_route]
                route.insert(best_pos, c)
                temp_dists[best_route] = route_dist(route)
                unassigned.remove(c)
            # Apply improvement
            new_routes, new_dists = improve(temp_routes, temp_dists)
            new_max = max(new_dists)
            # Simulated annealing acceptance with adaptive cooling
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = [route[:] for route in new_routes]
                report_best_vrp(best_routes)
                no_improve_count = 0
                pert_lower = 0.2
                pert_upper = 0.5
            else:
                delta = new_max - best_max
                if delta > 0 and temperature > 1e-6:
                    prob = math.exp(-delta / temperature)
                    if random.random() < prob:
                        routes = new_routes
                        route_dists = new_dists
                no_improve_count += 1
                # Adjust perturbation bounds after consecutive non-improvements
                if no_improve_count >= 3:
                    pert_lower = min(0.4, pert_lower + 0.05)
                    pert_upper = min(0.7, pert_upper + 0.05)
                # Adjust cooling factor
                if no_improve_count >= 3:
                    cooling_factor = cooling_factor_low
                else:
                    cooling_factor = cooling_factor_normal
                # Reheat after many non-improvements
                if no_improve_count >= reheat_threshold:
                    temperature = best_max * 0.2
                    no_improve_count = 0  # reset after reheat
            # Cool down
            temperature *= cooling_factor
    return best_routes