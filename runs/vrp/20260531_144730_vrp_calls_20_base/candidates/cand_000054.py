import numpy as np
import random
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    customers = list(range(1, n))
    INF = 1e15

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    def two_opt(route):
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_dist(new_route) < route_dist(best) - 1e-12:
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best

    # 1. Regret-2 construction with random tie-breaking
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = set(customers)

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
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    while unassigned:
        candidates = []
        for c in unassigned:
            best_val, best_route, best_pos, second_val = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            regret = second_val - best_val if second_val != INF else INF
            candidates.append((-regret, c, best_route, best_pos, best_val))
        max_regret = max(x[0] for x in candidates)
        top = [x for x in candidates if abs(x[0] - max_regret) < 1e-12]
        if random.random() < 0.1 and len(top) > 1:
            chosen = random.choice(top)
        else:
            top.sort(key=lambda x: x[1])
            chosen = top[0]
        _, c, best_route, best_pos, _ = chosen
        route = routes[best_route]
        route.insert(best_pos, c)
        route_dists[best_route] = route_dist(route)
        unassigned.remove(c)
        report_best_vrp(routes)

    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)

    for r_idx in range(truck_count):
        routes[r_idx] = two_opt(routes[r_idx])
        route_dists[r_idx] = route_dist(routes[r_idx])
        report_best_vrp(routes)
        cur_max = max(route_dists)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [route[:] for route in routes]

    max_iter = n * 10
    no_improve_streak = 0
    for iteration in range(max_iter):
        # Relocate from longest route
        max_dist_val = max(route_dists)
        max_idx = route_dists.index(max_dist_val)
        moved = False
        best_move = None
        best_new_max = max_dist_val
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
            routes[max_idx] = two_opt(routes[max_idx])
            routes[other_idx] = two_opt(routes[other_idx])
            route_dists[max_idx] = route_dist(routes[max_idx])
            route_dists[other_idx] = route_dist(routes[other_idx])
            report_best_vrp(routes)
            cur_max = max(route_dists)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [route[:] for route in routes]
                no_improve_streak = 0
            else:
                no_improve_streak += 1
            continue
        # Inter-route 2-opt* from longest route
        best_move = None
        best_new_max = max_dist_val
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
                        best_move = (i, other_idx, j, new_dist_max, new_dist_other)
        if best_move is not None:
            i, other_idx, j, new_dist_max, new_dist_other = best_move
            route_max = routes[max_idx]
            other_route = routes[other_idx]
            new_route_max = route_max[:i+1] + other_route[j+1:]
            new_route_other = other_route[:j+1] + route_max[i+1:]
            routes[max_idx] = new_route_max
            routes[other_idx] = new_route_other
            route_dists[max_idx] = route_dist(new_route_max)
            route_dists[other_idx] = route_dist(new_route_other)
            routes[max_idx] = two_opt(routes[max_idx])
            routes[other_idx] = two_opt(routes[other_idx])
            route_dists[max_idx] = route_dist(routes[max_idx])
            route_dists[other_idx] = route_dist(routes[other_idx])
            report_best_vrp(routes)
            cur_max = max(route_dists)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [route[:] for route in routes]
                no_improve_streak = 0
            else:
                no_improve_streak += 1
            continue
        # Ruin-recreate with probability 0.3
        if random.random() < 0.3 and len(customers) > 1:
            ruin_size = random.randint(1, max(1, len(customers) // 3))
            to_ruin = random.sample(customers, ruin_size)
            saved_routes = [r[:] for r in routes]
            saved_dists = route_dists[:]
            for cust in to_ruin:
                for ridx, route in enumerate(routes):
                    if cust in route:
                        new_route = [x for x in route if x != cust]
                        if len(new_route) < 2 or new_route[0] != 0:
                            new_route = [0] + new_route
                        if new_route[-1] != 0:
                            new_route = new_route + [0]
                        if len(new_route) == 2 and new_route[0] == 0 and new_route[1] == 0:
                            pass
                        elif len(new_route) == 2:
                            new_route = [0, 0]
                        routes[ridx] = new_route
                        route_dists[ridx] = route_dist(new_route)
                        break
            unrouted = set(to_ruin)
            while unrouted:
                candidates = []
                for c in unrouted:
                    best_val, best_route, best_pos, second_val = best_insertion(c, routes, route_dists)
                    if best_route == -1:
                        continue
                    regret = second_val - best_val if second_val != INF else INF
                    candidates.append((-regret, c, best_route, best_pos, best_val))
                if not candidates:
                    break
                max_regret = max(x[0] for x in candidates)
                top = [x for x in candidates if abs(x[0] - max_regret) < 1e-12]
                if random.random() < 0.1 and len(top) > 1:
                    chosen = random.choice(top)
                else:
                    top.sort(key=lambda x: x[1])
                    chosen = top[0]
                _, c, best_route, best_pos, _ = chosen
                route = routes[best_route]
                route.insert(best_pos, c)
                route_dists[best_route] = route_dist(route)
                unrouted.remove(c)
            cur_max = max(route_dists)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [route[:] for route in routes]
                no_improve_streak = 0
            else:
                routes = saved_routes
                route_dists = saved_dists
                no_improve_streak += 1
        else:
            no_improve_streak += 1
        # Restart via shaking (replaces DP split)
        if no_improve_streak >= max_iter // 5:
            # Shake: randomly relocate up to 30% of customers
            shake_size = random.randint(1, max(1, len(customers) // 3))
            to_move = random.sample(customers, shake_size)
            # Remove them
            for cust in to_move:
                for ridx, route in enumerate(routes):
                    if cust in route:
                        new_route = [x for x in route if x != cust]
                        if len(new_route) < 2 or new_route[0] != 0:
                            new_route = [0] + new_route
                        if new_route[-1] != 0:
                            new_route = new_route + [0]
                        if len(new_route) == 2 and new_route[0] == 0 and new_route[1] == 0:
                            pass
                        elif len(new_route) == 2:
                            new_route = [0, 0]
                        routes[ridx] = new_route
                        route_dists[ridx] = route_dist(new_route)
                        break
            # Reinsert using regret-2 with random tie-breaking
            unrouted = set(to_move)
            while unrouted:
                candidates = []
                for c in unrouted:
                    best_val, best_route, best_pos, second_val = best_insertion(c, routes, route_dists)
                    if best_route == -1:
                        continue
                    regret = second_val - best_val if second_val != INF else INF
                    candidates.append((-regret, c, best_route, best_pos, best_val))
                if not candidates:
                    break
                max_regret = max(x[0] for x in candidates)
                top = [x for x in candidates if abs(x[0] - max_regret) < 1e-12]
                if random.random() < 0.1 and len(top) > 1:
                    chosen = random.choice(top)
                else:
                    top.sort(key=lambda x: x[1])
                    chosen = top[0]
                _, c, best_route, best_pos, _ = chosen
                route = routes[best_route]
                route.insert(best_pos, c)
                route_dists[best_route] = route_dist(route)
                unrouted.remove(c)
            # Apply 2-opt to all routes
            for r_idx in range(truck_count):
                routes[r_idx] = two_opt(routes[r_idx])
                route_dists[r_idx] = route_dist(routes[r_idx])
            report_best_vrp(routes)
            cur_max = max(route_dists)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [route[:] for route in routes]
            no_improve_streak = 0

    return best_routes