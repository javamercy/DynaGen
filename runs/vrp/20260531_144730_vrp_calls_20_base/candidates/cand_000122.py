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

    # Random insertion initial construction (diverse)
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))
    random.shuffle(unassigned)
    for c in unassigned:
        best_candidates = []
        for r_idx in range(truck_count):
            for pos in range(1, len(routes[r_idx])):
                pred = routes[r_idx][pos-1]
                succ = routes[r_idx][pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = max(route_dists[r_idx], new_dist)
                best_candidates.append((new_max, r_idx, pos))
        best_candidates.sort(key=lambda x: x[0])
        # randomly choose among the top 3 (or all if fewer)
        top_k = min(3, len(best_candidates))
        candidate = random.choice(best_candidates[:top_k])
        r_idx = candidate[1]
        pos = candidate[2]
        routes[r_idx].insert(pos, c)
        route_dists[r_idx] = route_dist(routes[r_idx])

    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)
    best_total = total_dist(routes)
    report_best_vrp(best_routes)

    def improve(routes, route_dists):
        nonlocal best_routes, best_max, best_total
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
        max_iter = n * truck_count
        for _ in range(max_iter):
            improved_overall = False
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

    outer_iter = min(30, n * 2)
    stagnation_counter = 0
    num_removals_base = max(1, int((n-1) * 0.15))  # slightly more removals
    max_removals = max(1, int((n-1) * 0.45))
    for it in range(outer_iter):
        temperature = max(0.1, 100 * (1 - it / outer_iter) + 10)  # slower decay, min 0.1
        routes = [route[:] for route in best_routes]
        route_dists = [route_dist(r) for r in routes]
        extra = stagnation_counter * int((n-1) * 0.05)
        num_remove = min(max_removals, num_removals_base + extra)
        # Mix worst and random removal for diversity
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
        worst_candidates = removal_costs[:num_worst]
        random.shuffle(worst_candidates)
        # take some from worst, some random
        num_worst_remove = max(1, num_remove // 2)
        worst_removed = [c[1] for c in worst_candidates[:num_worst_remove]]
        remaining = [c for c in range(1, n) if c not in worst_removed]
        random.shuffle(remaining)
        num_random_remove = num_remove - len(worst_removed)
        random_removed = remaining[:num_random_remove]
        to_remove = list(set(worst_removed + random_removed))
        # ensure we don't remove too few
        if len(to_remove) < num_remove:
            additional = [c for c in range(1, n) if c not in to_remove]
            random.shuffle(additional)
            to_remove += additional[:num_remove - len(to_remove)]
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
        random.shuffle(unassigned)  # add randomness to insertion order
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
            # Deterministic tie-breaking: if temperature is high, still choose best regret but add small random? softmax already does that.
            selected = softmax_selection(items, temperature)
            c, best_route, best_pos, new_max = selected
            route = routes[best_route]
            route.insert(best_pos, c)
            route_dists[best_route] = route_dist(route)
            unassigned.remove(c)
        routes, route_dists = improve(routes, route_dists)
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
        # Perturbation: if stagnation > 3, apply random reinsertion of a few customers
        if stagnation_counter > 3:
            # Remove 10% random customers and reinsert with high temperature
            num_perturb = max(1, int((n-1)*0.1))
            perturb_candidates = list(range(1, n))
            random.shuffle(perturb_candidates)
            to_remove_perturb = perturb_candidates[:num_perturb]
            for c in to_remove_perturb:
                for r_idx in range(truck_count):
                    if c in routes[r_idx]:
                        pos = routes[r_idx].index(c)
                        pred = routes[r_idx][pos-1]
                        succ = routes[r_idx][pos+1]
                        route_dists[r_idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                        routes[r_idx].pop(pos)
                        break
            unassigned = to_remove_perturb[:]
            random.shuffle(unassigned)
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
                selected = softmax_selection(items, max(temperature, 50))  # high temperature
                c, best_route, best_pos, new_max = selected
                route = routes[best_route]
                route.insert(best_pos, c)
                route_dists[best_route] = route_dist(route)
                unassigned.remove(c)
            routes, route_dists = improve(routes, route_dists)
            cur_max = max(route_dists)
            cur_total = total_dist(routes)
            if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                best_max = cur_max
                best_total = cur_total
                best_routes = [route[:] for route in routes]
                report_best_vrp(best_routes)
                stagnation_counter = 0
            else:
                stagnation_counter = 0  # reset after perturbation
    return best_routes