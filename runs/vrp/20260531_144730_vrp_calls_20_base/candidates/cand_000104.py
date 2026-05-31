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
    restarts = max(5, min(10, n // 5))  # increased restarts for diversity
    # Parameters
    pert_lower = 0.3
    pert_upper = 0.6
    noise_scale_init = 0.05 * np.mean(distance_matrix[distance_matrix > 0]) if np.count_nonzero(distance_matrix) > 1 else 1.0
    cooling_factor_normal = 0.97
    cooling_factor_low = 0.92
    reheat_threshold = 4
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

        # Improvement function
        def improve(routes, route_dists):
            nonlocal best_routes, best_max, no_improve_count, pert_lower, pert_upper
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
                # Random relocate move (not always best)
                if random.random() < 0.3:  # 30% chance of random move for diversity
                    max_dist = max(route_dists)
                    max_idx = route_dists.index(max_dist)
                    route_max = routes[max_idx]
                    i = random.randint(1, len(route_max)-2)
                    c = route_max[i]
                    pred = route_max[i-1]
                    succ = route_max[i+1]
                    new_max_dist = route_dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
                    other_idx = random.randrange(truck_count)
                    while other_idx == max_idx:
                        other_idx = random.randrange(truck_count)
                    other_route = routes[other_idx]
                    pos = random.randint(1, len(other_route)-1)
                    pred_o = other_route[pos-1]
                    succ_o = other_route[pos]
                    new_other = route_dists[other_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                    other_max = 0.0
                    for j, d in enumerate(route_dists):
                        if j != max_idx and j != other_idx and d > other_max:
                            other_max = d
                    new_overall = max(other_max, new_max_dist, new_other)
                    if new_overall < best_max - 1e-12 or random.random() < 0.1:  # sometimes accept worse
                        route_max.pop(i)
                        other_route.insert(pos, c)
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
                            pert_lower = 0.3
                            pert_upper = 0.6
                        improved_overall = True
                else:
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
                            pert_lower = 0.3
                            pert_upper = 0.6
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
            pert_lower = 0.3
            pert_upper = 0.6

        # Outer loop with perturbations and simulated annealing acceptance
        outer_iterations = max(5, min(30, n // 3))  # more outer iterations
        temperature = best_max * 0.3
        cooling_factor = cooling_factor_normal
        for it in range(outer_iterations):
            # Start from best solution with some randomness
            if random.random() < 0.8:
                routes = [route[:] for route in best_routes]
                route_dists = [route_dist(r) for r in routes]
            else:
                routes = [route[:] for route in routes]
            # Dynamic perturbation size with higher range
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
            # Repair using regret with noise for diversity
            unassigned = to_remove[:]
            while unassigned:
                bests = []
                for c in unassigned:
                    best_new_max, best_route, best_pos, second_new_max = best_insertion(c, temp_routes, temp_dists, noise=True, noise_scale=noise_scale * 2)
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
                pert_lower = 0.3
                pert_upper = 0.6
            else:
                delta = new_max - best_max
                if delta > 0 and temperature > 1e-6:
                    prob = math.exp(-delta / temperature)
                    if random.random() < prob:
                        routes = new_routes
                        route_dists = new_dists
                else:
                    # Accept with small probability even if delta <=0? Actually delta>0 already
                    if random.random() < 0.05 and delta > 0:
                        routes = new_routes
                        route_dists = new_dists
                no_improve_count += 1
                # Adjust perturbation bounds after consecutive non-improvements
                if no_improve_count >= 2:
                    pert_lower = min(0.5, pert_lower + 0.05)
                    pert_upper = min(0.8, pert_upper + 0.05)
                # Adjust cooling factor
                if no_improve_count >= 3:
                    cooling_factor = cooling_factor_low
                else:
                    cooling_factor = cooling_factor_normal
                # Reheat more aggressively
                if no_improve_count >= reheat_threshold:
                    temperature = best_max * 0.3
                    no_improve_count = 0
                # Cool down
                temperature *= cooling_factor
    return best_routes