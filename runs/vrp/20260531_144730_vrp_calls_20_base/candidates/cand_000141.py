import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def total_dist(routes):
        return sum(route_dist(r) for r in routes)

    def best_insertion(c, routes, route_dists):
        # returns (new_max, best_route_idx, best_pos)
        best_max = float('inf')
        best_route = -1
        best_pos = -1
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
                if new_max < best_max:
                    best_max = new_max
                    best_route = r_idx
                    best_pos = pos
                elif abs(new_max - best_max) < 1e-12:
                    # tie-breaking: prefer smaller total distance? Actually we only have new_max, total not computed here.
                    # We'll keep the first found, but that's deterministic since we iterate routes in order.
                    pass
        return best_max, best_route, best_pos

    def construct(seed):
        random.seed(seed)
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = list(range(1, n))
        random.shuffle(unassigned)
        for c in unassigned:
            best_max, best_route, best_pos = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            routes[best_route].insert(best_pos, c)
            route_dists[best_route] = route_dist(routes[best_route])
        return routes, route_dists

    def improve(routes, route_dists):
        max_iter = n * truck_count
        for _ in range(max_iter):
            improved = False
            # relocate from longest route
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            route_max = routes[max_idx]
            best_move = None
            best_new_max = max_dist
            best_new_total = total_dist(routes)
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
                            best_new_total = total_dist(routes) - route_dists[max_idx] - route_dists[other_idx] + new_max_dist + new_other
                            best_move = (i, other_idx, pos, new_max_dist, new_other)
                        elif abs(new_overall - best_new_max) < 1e-12:
                            new_total = total_dist(routes) - route_dists[max_idx] - route_dists[other_idx] + new_max_dist + new_other
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
                improved = True
                # intra-route 2-opt on affected routes
                for r_idx in [max_idx, other_idx]:
                    for a in range(1, len(routes[r_idx])-2):
                        for b in range(a+1, len(routes[r_idx])-1):
                            old = distance_matrix[routes[r_idx][a-1], routes[r_idx][a]] + distance_matrix[routes[r_idx][b], routes[r_idx][b+1]]
                            new = distance_matrix[routes[r_idx][a-1], routes[r_idx][b]] + distance_matrix[routes[r_idx][a], routes[r_idx][b+1]]
                            if new < old - 1e-12:
                                routes[r_idx][a:b+1] = reversed(routes[r_idx][a:b+1])
                                route_dists[r_idx] = route_dist(routes[r_idx])
                continue

            # swap between longest and another
            best_swap = None
            best_new_max = max(route_dists)
            best_new_total = total_dist(routes)
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for i in range(1, len(route_max)-1):
                    c1 = route_max[i]
                    for j in range(1, len(other_route)-1):
                        c2 = other_route[j]
                        pred1 = route_max[i-1]
                        succ1 = route_max[i+1]
                        pred2 = other_route[j-1]
                        succ2 = other_route[j+1]
                        new_dist_max = route_dists[max_idx] - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                        new_dist_other = route_dists[other_idx] - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
                        other_max = 0.0
                        for k, d in enumerate(route_dists):
                            if k != max_idx and k != other_idx and d > other_max:
                                other_max = d
                        new_overall = max(other_max, new_dist_max, new_dist_other)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_new_total = total_dist(routes) - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
                        elif abs(new_overall - best_new_max) < 1e-12:
                            new_total = total_dist(routes) - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            if new_total < best_new_total - 1e-12:
                                best_new_max = new_overall
                                best_new_total = new_total
                                best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
            if best_swap is not None:
                i, other_idx, j, new_dist_max, new_dist_other = best_swap
                c1 = route_max[i]
                c2 = routes[other_idx][j]
                route_max[i] = c2
                routes[other_idx][j] = c1
                route_dists[max_idx] = new_dist_max
                route_dists[other_idx] = new_dist_other
                improved = True
                for r_idx in [max_idx, other_idx]:
                    for a in range(1, len(routes[r_idx])-2):
                        for b in range(a+1, len(routes[r_idx])-1):
                            old = distance_matrix[routes[r_idx][a-1], routes[r_idx][a]] + distance_matrix[routes[r_idx][b], routes[r_idx][b+1]]
                            new = distance_matrix[routes[r_idx][a-1], routes[r_idx][b]] + distance_matrix[routes[r_idx][a], routes[r_idx][b+1]]
                            if new < old - 1e-12:
                                routes[r_idx][a:b+1] = reversed(routes[r_idx][a:b+1])
                                route_dists[r_idx] = route_dist(routes[r_idx])
                continue

            # 2-opt* between longest and another
            best_cross = None
            best_new_max = max(route_dists)
            best_new_total = total_dist(routes)
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for i in range(1, len(route_max)-1):
                    for j in range(1, len(other_route)-1):
                        # ensure both end at depot: already satisfied by structure
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
                            best_new_total = total_dist(routes) - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            best_cross = (i, other_idx, j, new_dist_max, new_dist_other)
                        elif abs(new_overall - best_new_max) < 1e-12:
                            new_total = total_dist(routes) - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                            if new_total < best_new_total - 1e-12:
                                best_new_max = new_overall
                                best_new_total = new_total
                                best_cross = (i, other_idx, j, new_dist_max, new_dist_other)
            if best_cross is not None:
                i, other_idx, j, new_dist_max, new_dist_other = best_cross
                new_route_max = route_max[:i+1] + routes[other_idx][j+1:]
                new_route_other = routes[other_idx][:j+1] + route_max[i+1:]
                routes[max_idx] = new_route_max
                routes[other_idx] = new_route_other
                route_dists[max_idx] = route_dist(new_route_max)
                route_dists[other_idx] = route_dist(new_route_other)
                improved = True
                for r_idx in [max_idx, other_idx]:
                    for a in range(1, len(routes[r_idx])-2):
                        for b in range(a+1, len(routes[r_idx])-1):
                            old = distance_matrix[routes[r_idx][a-1], routes[r_idx][a]] + distance_matrix[routes[r_idx][b], routes[r_idx][b+1]]
                            new = distance_matrix[routes[r_idx][a-1], routes[r_idx][b]] + distance_matrix[routes[r_idx][a], routes[r_idx][b+1]]
                            if new < old - 1e-12:
                                routes[r_idx][a:b+1] = reversed(routes[r_idx][a:b+1])
                                route_dists[r_idx] = route_dist(routes[r_idx])
                continue

            if not improved:
                break
        return routes, route_dists

    best_routes = None
    best_max = float('inf')
    best_total = float('inf')
    num_restarts = min(20, 5 + n // 10)
    for restart in range(num_restarts):
        seed = restart  # deterministic
        routes, route_dists = construct(seed)
        routes, route_dists = improve(routes, route_dists)
        cur_max = max(route_dists)
        cur_total = total_dist(routes)
        if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
            best_max = cur_max
            best_total = cur_total
            best_routes = [route[:] for route in routes]
            report_best_vrp(best_routes)
    return best_routes if best_routes is not None else []