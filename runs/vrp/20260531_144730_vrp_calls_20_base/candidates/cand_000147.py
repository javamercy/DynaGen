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

    def softmax_selection(items, temperature):
        if temperature < 1e-12:
            items.sort(key=lambda x: x[0])
            return items[0][1]
        values = [v for v, _ in items]
        max_val = max(values)
        shifted = [(max_val - v) / temperature for v in values]
        max_shift = max(shifted) if shifted else 0
        shifted = [s - max_shift for s in shifted]
        exp_vals = [np.exp(s) for s in shifted]
        total_exp = sum(exp_vals)
        if total_exp == 0:
            return random.choice(items)[1]
        probs = [e / total_exp for e in exp_vals]
        r = random.random()
        cumulative = 0.0
        for i, prob in enumerate(probs):
            cumulative += prob
            if r <= cumulative:
                return items[i][1]
        return items[-1][1]

    def construct_solution(use_random=False):
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
                if use_random:
                    bests.append((random.random(), c, best_route, best_pos, best_new_max))
                else:
                    bests.append((-regret, c, best_route, best_pos, best_new_max))
            bests.sort(key=lambda x: (x[0], x[1]))
            _, c, best_route, best_pos, new_max = bests[0]
            route = routes[best_route]
            route.insert(best_pos, c)
            route_dists[best_route] = route_dist(route)
            unassigned.remove(c)
        return routes, route_dists

    def improve_solution(routes, route_dists):
        nonlocal best_routes, best_max, best_total
        # Intra-route 2-opt (limited iterations)
        for r_idx in range(truck_count):
            improved = True
            max_iter = n
            while improved and max_iter > 0:
                improved = False
                max_iter -= 1
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
        # Inter-route improvement (best-improvement relocate, swap, 2-opt*)
        max_iter = n * 2
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
                        if new_overall < best_new_max - 1e-12 or (abs(new_overall - best_new_max) < 1e-12 and new_total < best_new_total - 1e-12):
                            best_new_max = new_overall
                            best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_max_dist + new_other
                            best_move = (i, other_idx, pos, new_max_dist, new_other)
            if best_move is not None:
                i, other_idx, pos, new_max_dist, new_other = best_move
                c = route_max.pop(i)
                routes[other_idx].insert(pos, c)
                route_dists[max_idx] = new_max_dist
                route_dists[other_idx] = new_other
                improved_overall = True
            else:
                # Best swap
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
                            pred1 = route_max[i-1]
                            succ1 = route_max[i+1]
                            new_dist_max = route_dists[max_idx] - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                            pred2 = other_route[j-1]
                            succ2 = other_route[j+1]
                            new_dist_other = route_dists[other_idx] - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
                            other_max = 0.0
                            for k, d in enumerate(route_dists):
                                if k != max_idx and k != other_idx and d > other_max:
                                    other_max = d
                            new_overall = max(other_max, new_dist_max, new_dist_other)
                            if new_overall < best_new_max - 1e-12 or (abs(new_overall - best_new_max) < 1e-12 and new_total < best_new_total - 1e-12):
                                best_new_max = new_overall
                                best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                                best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
                if best_swap is not None:
                    i, other_idx, j, new_dist_max, new_dist_other = best_swap
                    c1 = route_max[i]
                    c2 = routes[other_idx][j]
                    route_max[i] = c2
                    routes[other_idx][j] = c1
                    route_dists[max_idx] = new_dist_max
                    route_dists[other_idx] = new_dist_other
                    improved_overall = True
                else:
                    # Best 2-opt*
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
                                if new_overall < best_new_max - 1e-12 or (abs(new_overall - best_new_max) < 1e-12 and new_total < best_new_total - 1e-12):
                                    best_new_max = new_overall
                                    best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                                    best_cross = (i, other_idx, j, new_dist_max, new_dist_other)
                    if best_cross is not None:
                        i, other_idx, j, new_dist_max, new_dist_other = best_cross
                        new_route_max = route_max[:i+1] + routes[other_idx][j+1:]
                        new_route_other = routes[other_idx][:j+1] + route_max[i+1:]
                        routes[max_idx] = new_route_max
                        routes[other_idx] = new_route_other
                        route_dists[max_idx] = route_dist(new_route_max)
                        route_dists[other_idx] = route_dist(new_route_other)
                        improved_overall = True
            if improved_overall:
                # Intra 2-opt on affected routes
                for r_idx in [max_idx, other_idx]:
                    improved = True
                    max_inner = n
                    while improved and max_inner > 0:
                        improved = False
                        max_inner -= 1
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
            else:
                break
        return routes, route_dists

    # Initial construction
    routes, route_dists = construct_solution(use_random=False)
    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)
    best_total = total_dist(routes)
    report_best_vrp(best_routes)

    # Initial improvement
    routes, route_dists = improve_solution(routes, route_dists)

    # Adaptive LNS outer loop with exploration
    outer_iter = min(15, n * 2)
    stagnation_counter = 0
    num_removals_base = max(1, int((n-1) * 0.1))
    max_removals = max(1, int((n-1) * 0.5))
    for it in range(outer_iter):
        temperature = max(0, 200 * (1 - it / outer_iter))
        if stagnation_counter >= 5 and it > 10:
            routes, route_dists = construct_solution(use_random=True)
            routes, route_dists = improve_solution(routes, route_dists)
            cur_max = max(route_dists)
            cur_total = total_dist(routes)
            if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                best_max = cur_max
                best_total = cur_total
                best_routes = [route[:] for route in routes]
                report_best_vrp(best_routes)
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            continue

        routes = [route[:] for route in best_routes]
        route_dists = [route_dist(r) for r in routes]
        extra = stagnation_counter * int((n-1) * 0.05)
        num_remove = min(max_removals, num_removals_base + extra)
        # Mixed removal: half worst, half random
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
        half = max(1, num_remove // 2)
        candidates = removal_costs[:max(half, len(removal_costs)//2)]
        random.shuffle(candidates)
        to_remove = [c[1] for c in candidates[:half]]
        remaining = [x for x in removal_costs if x[1] not in to_remove]
        random.shuffle(remaining)
        to_remove += [x[1] for x in remaining[:num_remove - len(to_remove)]]
        for c in to_remove:
            for r_idx in range(truck_count):
                if c in routes[r_idx]:
                    pos = routes[r_idx].index(c)
                    pred = routes[r_idx][pos-1]
                    succ = routes[r_idx][pos+1]
                    route_dists[r_idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                    routes[r_idx].pop(pos)
                    break
        # Repair with softmax regret
        unassigned = to_remove[:]
        while unassigned:
            items = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
                if best_route == -1:
                    continue
                regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                items.append((-regret, (c, best_route, best_pos, best_new_max)))
            if not items:
                break
            selected = softmax_selection(items, temperature)
            c, best_route, best_pos, new_max = selected
            route = routes[best_route]
            route.insert(best_pos, c)
            route_dists[best_route] = route_dist(route)
            unassigned.remove(c)
        # Local search
        routes, route_dists = improve_solution(routes, route_dists)
        cur_max = max(route_dists)
        cur_total = total_dist(routes)
        if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
            best_max = cur_max
            best_total = cur_total
            best_routes = [route[:] for route in routes]
            report_best_vrp(best_routes)
            stagnation_counter = 0
        else:
            stagnation_counter += 1
    return best_routes