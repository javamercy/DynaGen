import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)
    
    # Initial routes
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))

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
    current_routes = [route[:] for route in routes]
    current_dists = route_dists[:]

    # Intra-route 2-opt on initial solution
    for r_idx in range(truck_count):
        improved = True
        while improved:
            improved = False
            route = current_routes[r_idx]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new_dist = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new_dist < old - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        current_dists[r_idx] = route_dist(route)
                        break
                if improved:
                    break
        cur_max = max(current_dists)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [route[:] for route in current_routes]
            report_best_vrp(best_routes)

    # LNS iterations (limited)
    max_lns_iter = n * 3
    for _ in range(max_lns_iter):
        # Destroy: remove random subset (10%-40%)
        num_remove = random.randint(max(1, (n-1)//10), max(1, (n-1)*4//10))
        customers = list(range(1, n))
        random.shuffle(customers)
        to_remove = customers[:num_remove]
        temp_routes = [route[:] for route in current_routes]
        temp_dists = current_dists[:]
        for c in to_remove:
            for r_idx in range(truck_count):
                if c in temp_routes[r_idx]:
                    pos = temp_routes[r_idx].index(c)
                    pred = temp_routes[r_idx][pos-1]
                    succ = temp_routes[r_idx][pos+1]
                    temp_dists[r_idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                    temp_routes[r_idx].pop(pos)
                    break
        # Repair using regret
        unassigned = to_remove[:]
        while unassigned:
            bests = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion(c, temp_routes, temp_dists)
                if best_route == -1:
                    continue
                regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                bests.append((-regret, c, best_route, best_pos, best_new_max))
            bests.sort(key=lambda x: (x[0], x[1]))
            _, c, best_route, best_pos, new_max = bests[0]
            route = temp_routes[best_route]
            route.insert(best_pos, c)
            temp_dists[best_route] = route_dist(route)
            unassigned.remove(c)
        new_max = max(temp_dists)
        if new_max < best_max - 1e-12:
            best_max = new_max
            best_routes = [route[:] for route in temp_routes]
            report_best_vrp(best_routes)
            current_routes = [route[:] for route in temp_routes]
            current_dists = temp_dists[:]

    # Intensive local search (focus on longest route)
    max_iter = n * truck_count
    for _ in range(max_iter):
        improved_overall = False
        # Best-improvement relocate from longest route
        max_dist = max(current_dists)
        max_idx = current_dists.index(max_dist)
        best_move = None
        best_new_max = max_dist
        route_max = current_routes[max_idx]
        for i in range(1, len(route_max)-1):
            c = route_max[i]
            pred = route_max[i-1]
            succ = route_max[i+1]
            new_max_dist = current_dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = current_routes[other_idx]
                for pos in range(1, len(other_route)):
                    pred_o = other_route[pos-1]
                    succ_o = other_route[pos]
                    new_other = current_dists[other_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                    other_max_val = 0.0
                    for j, d in enumerate(current_dists):
                        if j != max_idx and j != other_idx and d > other_max_val:
                            other_max_val = d
                    new_overall = max(other_max_val, new_max_dist, new_other)
                    if new_overall < best_new_max - 1e-12:
                        best_new_max = new_overall
                        best_move = (i, other_idx, pos, new_max_dist, new_other)
        if best_move is not None:
            i, other_idx, pos, new_max_dist, new_other = best_move
            c = route_max.pop(i)
            current_routes[other_idx].insert(pos, c)
            current_dists[max_idx] = new_max_dist
            current_dists[other_idx] = new_other
            for r_idx in [max_idx, other_idx]:
                improved = True
                while improved:
                    improved = False
                    route = current_routes[r_idx]
                    for a in range(1, len(route)-2):
                        for b in range(a+1, len(route)-1):
                            old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                            new_d = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                            if new_d < old - 1e-12:
                                route[a:b+1] = reversed(route[a:b+1])
                                improved = True
                                current_dists[r_idx] = route_dist(route)
                                break
                        if improved:
                            break
            cur_max = max(current_dists)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [route[:] for route in current_routes]
                report_best_vrp(best_routes)
            improved_overall = True

        # Best-improvement swap
        if not improved_overall:
            max_dist = max(current_dists)
            max_idx = current_dists.index(max_dist)
            best_swap = None
            best_new_max = max_dist
            route_max = current_routes[max_idx]
            for i in range(1, len(route_max)-1):
                c1 = route_max[i]
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = current_routes[other_idx]
                    for j in range(1, len(other_route)-1):
                        c2 = other_route[j]
                        pred1 = route_max[i-1]
                        succ1 = route_max[i+1]
                        new_dist_max = current_dists[max_idx] - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                        pred2 = other_route[j-1]
                        succ2 = other_route[j+1]
                        new_dist_other = current_dists[other_idx] - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
                        other_max_val = 0.0
                        for k, d in enumerate(current_dists):
                            if k != max_idx and k != other_idx and d > other_max_val:
                                other_max_val = d
                        new_overall = max(other_max_val, new_dist_max, new_dist_other)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
            if best_swap is not None:
                i, other_idx, j, new_dist_max, new_dist_other = best_swap
                c1 = route_max[i]
                c2 = current_routes[other_idx][j]
                route_max[i] = c2
                current_routes[other_idx][j] = c1
                current_dists[max_idx] = new_dist_max
                current_dists[other_idx] = new_dist_other
                for r_idx in [max_idx, other_idx]:
                    improved = True
                    while improved:
                        improved = False
                        route = current_routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new_d = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new_d < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved = True
                                    current_dists[r_idx] = route_dist(route)
                                    break
                            if improved:
                                break
                cur_max = max(current_dists)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [route[:] for route in current_routes]
                    report_best_vrp(best_routes)
                improved_overall = True

        # Best-improvement 2-opt*
        if not improved_overall:
            max_dist = max(current_dists)
            max_idx = current_dists.index(max_dist)
            best_cross = None
            best_new_max = max_dist
            route_max = current_routes[max_idx]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = current_routes[other_idx]
                for i in range(1, len(route_max)-1):
                    for j in range(1, len(other_route)-1):
                        if route_max[-1] != 0 or other_route[-1] != 0:
                            continue
                        old1 = distance_matrix[route_max[i], route_max[i+1]]
                        old2 = distance_matrix[other_route[j], other_route[j+1]]
                        new1 = distance_matrix[route_max[i], other_route[j+1]]
                        new2 = distance_matrix[other_route[j], route_max[i+1]]
                        new_dist_max = current_dists[max_idx] - old1 + new1
                        new_dist_other = current_dists[other_idx] - old2 + new2
                        other_max_val = 0.0
                        for k, d in enumerate(current_dists):
                            if k != max_idx and k != other_idx and d > other_max_val:
                                other_max_val = d
                        new_overall = max(other_max_val, new_dist_max, new_dist_other)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_cross = (i, other_idx, j, new_dist_max, new_dist_other)
            if best_cross is not None:
                i, other_idx, j, new_dist_max, new_dist_other = best_cross
                new_route_max = route_max[:i+1] + other_route[j+1:]
                new_route_other = other_route[:j+1] + route_max[i+1:]
                current_routes[max_idx] = new_route_max
                current_routes[other_idx] = new_route_other
                current_dists[max_idx] = route_dist(new_route_max)
                current_dists[other_idx] = route_dist(new_route_other)
                for r_idx in [max_idx, other_idx]:
                    improved = True
                    while improved:
                        improved = False
                        route = current_routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new_d = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new_d < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved = True
                                    current_dists[r_idx] = route_dist(route)
                                    break
                            if improved:
                                break
                cur_max = max(current_dists)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [route[:] for route in current_routes]
                    report_best_vrp(best_routes)
                improved_overall = True

        if not improved_overall:
            break

    return best_routes