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

    def evaluate_insertion(c, route, route_dist, position):
        pred = route[position-1]
        succ = route[position]
        new_dist = route_dist - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
        return new_dist

    def best_insertion(c, routes, route_dists):
        best_new_max = float('inf')
        best_route = -1
        best_pos = -1
        second_new_max = float('inf')
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            other_max = max(route_dists[j] for j in range(truck_count) if j != r_idx) if truck_count > 1 else 0.0
            for pos in range(1, len(route)):
                new_dist = evaluate_insertion(c, route, route_dists[r_idx], pos)
                new_max = max(other_max, new_dist)
                if new_max < best_new_max:
                    second_new_max = best_new_max
                    best_new_max = new_max
                    best_route = r_idx
                    best_pos = pos
                elif new_max < second_new_max:
                    second_new_max = new_max
        return best_new_max, best_route, best_pos, second_new_max

    # Initial construction with regret-2
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))
    while unassigned:
        best_items = []
        for c in unassigned:
            best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
            best_items.append((-regret, c, best_route, best_pos, best_new_max))
        best_items.sort(key=lambda x: (x[0], x[1]))
        _, c, best_route, best_pos, _ = best_items[0]
        route = routes[best_route]
        route.insert(best_pos, c)
        route_dists[best_route] = route_dist(route)
        unassigned.remove(c)

    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)
    best_total = total_dist(routes)
    report_best_vrp(best_routes)

    # Local search
    def local_search(routes, route_dists):
        nonlocal best_routes, best_max, best_total
        max_iter = n * truck_count
        for _ in range(max_iter):
            improved = False
            # Best relocate from max-dist route
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
                new_dist_max = route_dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    other_max = max(route_dists[j] for j in range(truck_count) if j not in (max_idx, other_idx)) if truck_count > 2 else 0.0
                    for pos in range(1, len(other_route)):
                        pred_o = other_route[pos-1]
                        succ_o = other_route[pos]
                        new_dist_other = route_dists[other_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                        candidate_max = max(other_max, new_dist_max, new_dist_other)
                        if candidate_max < best_new_max - 1e-12:
                            best_new_max = candidate_max
                            best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            best_move = (max_idx, i, other_idx, pos, new_dist_max, new_dist_other)
                        elif abs(candidate_max - best_new_max) < 1e-12:
                            candidate_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            if candidate_total < best_new_total - 1e-12:
                                best_new_max = candidate_max
                                best_new_total = candidate_total
                                best_move = (max_idx, i, other_idx, pos, new_dist_max, new_dist_other)
            if best_move is not None:
                src_idx, i, dst_idx, pos, new_dist_src, new_dist_dst = best_move
                c = routes[src_idx].pop(i)
                routes[dst_idx].insert(pos, c)
                route_dists[src_idx] = new_dist_src
                route_dists[dst_idx] = new_dist_dst
                # Intra-route 2-opt on affected routes
                for r_idx in (src_idx, dst_idx):
                    improved_intra = True
                    while improved_intra:
                        improved_intra = False
                        route = routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved_intra = True
                                    route_dists[r_idx] = route_dist(route)
                                    break
                            if improved_intra:
                                break
                cur_max = max(route_dists)
                cur_total = total_dist(routes)
                if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [route[:] for route in routes]
                    report_best_vrp(best_routes)
                improved = True
                continue

            # Best swap (inter-route)
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
                        other_max = max(route_dists[k] for k in range(truck_count) if k not in (max_idx, other_idx)) if truck_count > 2 else 0.0
                        candidate_max = max(other_max, new_dist_max, new_dist_other)
                        if candidate_max < best_new_max - 1e-12:
                            best_new_max = candidate_max
                            best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            best_swap = (max_idx, i, other_idx, j, new_dist_max, new_dist_other)
                        elif abs(candidate_max - best_new_max) < 1e-12:
                            candidate_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            if candidate_total < best_new_total - 1e-12:
                                best_new_max = candidate_max
                                best_new_total = candidate_total
                                best_swap = (max_idx, i, other_idx, j, new_dist_max, new_dist_other)
            if best_swap is not None:
                src_idx, i, dst_idx, j, new_dist_src, new_dist_dst = best_swap
                routes[src_idx][i], routes[dst_idx][j] = routes[dst_idx][j], routes[src_idx][i]
                route_dists[src_idx] = new_dist_src
                route_dists[dst_idx] = new_dist_dst
                for r_idx in (src_idx, dst_idx):
                    improved_intra = True
                    while improved_intra:
                        improved_intra = False
                        route = routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved_intra = True
                                    route_dists[r_idx] = route_dist(route)
                                    break
                            if improved_intra:
                                break
                cur_max = max(route_dists)
                cur_total = total_dist(routes)
                if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [route[:] for route in routes]
                    report_best_vrp(best_routes)
                improved = True
                continue

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
                        if route_max[-1] != 0 or other_route[-1] != 0:
                            continue
                        old1 = distance_matrix[route_max[i], route_max[i+1]]
                        old2 = distance_matrix[other_route[j], other_route[j+1]]
                        new1 = distance_matrix[route_max[i], other_route[j+1]]
                        new2 = distance_matrix[other_route[j], route_max[i+1]]
                        new_dist_max = route_dists[max_idx] - old1 + new1
                        new_dist_other = route_dists[other_idx] - old2 + new2
                        other_max = max(route_dists[k] for k in range(truck_count) if k not in (max_idx, other_idx)) if truck_count > 2 else 0.0
                        candidate_max = max(other_max, new_dist_max, new_dist_other)
                        if candidate_max < best_new_max - 1e-12:
                            best_new_max = candidate_max
                            best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            best_cross = (max_idx, i, other_idx, j, new_dist_max, new_dist_other)
                        elif abs(candidate_max - best_new_max) < 1e-12:
                            candidate_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            if candidate_total < best_new_total - 1e-12:
                                best_new_max = candidate_max
                                best_new_total = candidate_total
                                best_cross = (max_idx, i, other_idx, j, new_dist_max, new_dist_other)
            if best_cross is not None:
                src_idx, i, dst_idx, j, new_dist_src, new_dist_dst = best_cross
                new_route_src = routes[src_idx][:i+1] + routes[dst_idx][j+1:]
                new_route_dst = routes[dst_idx][:j+1] + routes[src_idx][i+1:]
                routes[src_idx] = new_route_src
                routes[dst_idx] = new_route_dst
                route_dists[src_idx] = new_dist_src
                route_dists[dst_idx] = new_dist_dst
                for r_idx in (src_idx, dst_idx):
                    improved_intra = True
                    while improved_intra:
                        improved_intra = False
                        route = routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved_intra = True
                                    route_dists[r_idx] = route_dist(route)
                                    break
                            if improved_intra:
                                break
                cur_max = max(route_dists)
                cur_total = total_dist(routes)
                if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                    best_max = cur_max
                    best_total = cur_total
                    best_routes = [route[:] for route in routes]
                    report_best_vrp(best_routes)
                improved = True

            if not improved:
                break
        return routes, route_dists

    routes, route_dists = local_search(routes, route_dists)

    # Ruin-and-recreate iterations
    outer_iter = min(10, n)
    num_removals = max(1, int((n-1) * 0.2))
    for _ in range(outer_iter):
        routes = [route[:] for route in best_routes]
        route_dists = [route_dist(r) for r in routes]
        # Worst removal (select customers with highest saving if removed)
        removal_candidates = []
        for c in range(1, n):
            for r_idx, route in enumerate(routes):
                if c in route:
                    pos = route.index(c)
                    pred = route[pos-1]
                    succ = route[pos+1]
                    saving = distance_matrix[pred, c] + distance_matrix[c, succ] - distance_matrix[pred, succ]
                    removal_candidates.append((saving, c, r_idx))
                    break
        removal_candidates.sort(key=lambda x: -x[0])
        to_remove = [item[1] for item in removal_candidates[:num_removals]]
        for c in to_remove:
            for r_idx in range(truck_count):
                if c in routes[r_idx]:
                    pos = routes[r_idx].index(c)
                    pred = routes[r_idx][pos-1]
                    succ = routes[r_idx][pos+1]
                    route_dists[r_idx] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                    routes[r_idx].pop(pos)
                    break
        # Repair with regret-2
        unassigned = to_remove[:]
        while unassigned:
            best_items = []
            for c in unassigned:
                best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
                if best_route == -1:
                    continue
                regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
                best_items.append((-regret, c, best_route, best_pos, best_new_max))
            best_items.sort(key=lambda x: (x[0], x[1]))
            _, c, best_route, best_pos, _ = best_items[0]
            route = routes[best_route]
            route.insert(best_pos, c)
            route_dists[best_route] = route_dist(route)
            unassigned.remove(c)
        # Local search
        routes, route_dists = local_search(routes, route_dists)
        cur_max = max(route_dists)
        cur_total = total_dist(routes)
        if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
            best_max = cur_max
            best_total = cur_total
            best_routes = [route[:] for route in routes]
            report_best_vrp(best_routes)

    return best_routes