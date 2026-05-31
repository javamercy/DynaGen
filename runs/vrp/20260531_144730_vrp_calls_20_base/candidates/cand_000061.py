import numpy as np
import random
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)
    INF = 1e15

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(c, routes, route_dists):
        best = (INF, -1, -1)
        second = (INF, -1, -1)
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
                if new_max < best[0] - 1e-12:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0] - 1e-12:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    # Construction
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))
    while unassigned:
        bests = []
        for c in unassigned:
            best_val, best_route, best_pos, second_val = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            regret = second_val - best_val if second_val != INF else INF
            bests.append((-regret, c, best_route, best_pos, best_val))
        # deterministic tie-breaking: if regrets equal, choose smaller customer index
        bests.sort(key=lambda x: (x[0], x[1]))
        _, c, best_route, best_pos, _ = bests[0]
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
                    new_dist = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new_dist < old - 1e-12:
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

    # Local search loop: apply relocate, swap, 2-opt* on longest route
    max_iter_local = n * 2
    for _ in range(max_iter_local):
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
            # Intra-2-opt on modified routes
            for r_idx in [max_idx, other_idx]:
                improved = True
                while improved:
                    improved = False
                    route = routes[r_idx]
                    for a in range(1, len(route)-2):
                        for b in range(a+1, len(route)-1):
                            old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                            new_dist = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                            if new_dist < old - 1e-12:
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

        # Best-improvement swap if no relocate
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
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
            if best_swap is not None:
                i, other_idx, j, new_dist_max, new_dist_other = best_swap
                c1 = route_max[i]
                c2 = routes[other_idx][j]
                route_max[i] = c2
                routes[other_idx][j] = c1
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
                                new_dist = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new_dist < old - 1e-12:
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

        # Best-improvement 2-opt* if no swap
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
                                new_dist = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new_dist < old - 1e-12:
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

    # LNS phase
    lns_iter = min(5, max(1, n // 20))
    for _ in range(lns_iter):
        # Destroy: remove 30-50% of customers
        customers = list(range(1, n))
        random.shuffle(customers)
        num_remove = max(1, len(customers) * random.randint(30, 50) // 100)
        to_remove = customers[:num_remove]
        temp_routes = [route[:] for route in best_routes]
        temp_dists = [route_dist(r) for r in temp_routes]
        for c in to_remove:
            for r_idx in range(truck_count):
                if c in temp_routes[r_idx]:
                    pos = temp_routes[r_idx].index(c)
                    pred = temp_routes[r_idx][pos-1]
                    succ = temp_routes[r_idx][pos+1]
                    temp_dists[r_idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                    temp_routes[r_idx].pop(pos)
                    break
        # Repair: regret-2
        unassigned = to_remove[:]
        while unassigned:
            bests = []
            for c in unassigned:
                best_val, best_route, best_pos, second_val = best_insertion(c, temp_routes, temp_dists)
                if best_route == -1:
                    continue
                regret = second_val - best_val if second_val != INF else INF
                bests.append((-regret, c, best_route, best_pos, best_val))
            bests.sort(key=lambda x: (x[0], x[1]))
            _, c, best_route, best_pos, _ = bests[0]
            temp_routes[best_route].insert(best_pos, c)
            temp_dists[best_route] = route_dist(temp_routes[best_route])
            unassigned.remove(c)
        new_max = max(temp_dists)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [route[:] for route in temp_routes]
            report_best_vrp(best_routes)
            # re-apply local search
            routes = [route[:] for route in best_routes]
            route_dists = [route_dist(r) for r in routes]
            for _ in range(max_iter_local):
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
                                    new_dist = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                    if new_dist < old - 1e-12:
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
                                if new_overall < best_new_max - 1e-12:
                                    best_new_max = new_overall
                                    best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
                    if best_swap is not None:
                        i, other_idx, j, new_dist_max, new_dist_other = best_swap
                        c1 = route_max[i]
                        c2 = routes[other_idx][j]
                        route_max[i] = c2
                        routes[other_idx][j] = c1
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
                                        new_dist = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                        if new_dist < old - 1e-12:
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
                # 2-opt*
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
                                        new_dist = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                        if new_dist < old - 1e-12:
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
        else:
            # If no improvement, try restart from best with perturbation
            if random.random() < 0.5:
                # Perturb: relocate 10% of customers randomly
                routes = [route[:] for route in best_routes]
                route_dists = [route_dist(r) for r in routes]
                num_perturb = max(1, (n-1) // 10)
                for _ in range(num_perturb):
                    # pick a random customer from a random route that has at least one customer
                    nonempty = [r for r in range(truck_count) if len(routes[r]) > 2]
                    if not nonempty:
                        break
                    src = random.choice(nonempty)
                    src_route = routes[src]
                    if len(src_route) < 3:
                        continue
                    i = random.randint(1, len(src_route)-2)
                    c = src_route.pop(i)
                    # choose a destination route (may be same but different position, or different)
                    dst = random.choice(range(truck_count))
                    dst_route = routes[dst]
                    pos = random.randint(1, len(dst_route)-1)
                    dst_route.insert(pos, c)
                # update distances and run local search
                route_dists = [route_dist(r) for r in routes]
                for r_idx in range(truck_count):
                    improved = True
                    while improved:
                        improved = False
                        route = routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new_dist = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new_dist < old - 1e-12:
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

    return best_routes