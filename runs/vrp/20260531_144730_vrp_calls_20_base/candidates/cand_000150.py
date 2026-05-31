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

    def construct_greedy():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = list(range(1, n))
        random.shuffle(unassigned)
        for c in unassigned:
            best_new_max = float('inf')
            best_truck = -1
            best_pos = -1
            for t in range(truck_count):
                r = routes[t]
                for pos in range(1, len(r)):
                    pred = r[pos-1]
                    succ = r[pos]
                    new_dist = route_dists[t] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                    other_max = 0.0
                    for k in range(truck_count):
                        if k != t and route_dists[k] > other_max:
                            other_max = route_dists[k]
                    overall_max = max(other_max, new_dist)
                    if overall_max < best_new_max - 1e-12:
                        best_new_max = overall_max
                        best_truck = t
                        best_pos = pos
            if best_truck != -1:
                routes[best_truck].insert(best_pos, c)
                route_dists[best_truck] = route_dist(routes[best_truck])
        return routes, route_dists

    def improve(routes, route_dists, best_routes, best_max, best_total):
        for t in range(truck_count):
            r = routes[t]
            improved = True
            while improved:
                improved = False
                for i in range(1, len(r)-2):
                    for j in range(i+1, len(r)-1):
                        old = distance_matrix[r[i-1], r[i]] + distance_matrix[r[j], r[j+1]]
                        new = distance_matrix[r[i-1], r[j]] + distance_matrix[r[i], r[j+1]]
                        if new < old - 1e-12:
                            r[i:j+1] = reversed(r[i:j+1])
                            route_dists[t] = route_dist(r)
                            improved = True
                            break
                    if improved:
                        break
        max_iter = n * truck_count * 2
        for _ in range(max_iter):
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_move = None
            best_new_max = max_dist
            for i in range(1, len(routes[max_idx])-1):
                c = routes[max_idx][i]
                new_max_dist = route_dists[max_idx] - distance_matrix[routes[max_idx][i-1], c] - distance_matrix[c, routes[max_idx][i+1]] + distance_matrix[routes[max_idx][i-1], routes[max_idx][i+1]]
                for t in range(truck_count):
                    if t == max_idx:
                        continue
                    r = routes[t]
                    for pos in range(1, len(r)):
                        pred = r[pos-1]
                        succ = r[pos]
                        new_dist_other = route_dists[t] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                        other_max = 0.0
                        for k in range(truck_count):
                            if k != max_idx and k != t and route_dists[k] > other_max:
                                other_max = route_dists[k]
                        overall_max = max(other_max, new_max_dist, new_dist_other)
                        if overall_max < best_new_max - 1e-12:
                            best_new_max = overall_max
                            best_move = (max_idx, i, t, pos, new_max_dist, new_dist_other)
                        elif abs(overall_max - best_new_max) < 1e-12:
                            cur_total = total_dist(routes)
                            new_total = cur_total - route_dists[max_idx] - route_dists[t] + new_max_dist + new_dist_other
                            if new_total < cur_total - 1e-12:
                                best_new_max = overall_max
                                best_move = (max_idx, i, t, pos, new_max_dist, new_dist_other)
            if best_move is not None:
                max_idx, i, t, pos, new_max_dist, new_dist_other = best_move
                c = routes[max_idx].pop(i)
                routes[t].insert(pos, c)
                route_dists[max_idx] = new_max_dist
                route_dists[t] = new_dist_other
                for idx in [max_idx, t]:
                    r = routes[idx]
                    improved = True
                    while improved:
                        improved = False
                        for a in range(1, len(r)-2):
                            for b in range(a+1, len(r)-1):
                                old = distance_matrix[r[a-1], r[a]] + distance_matrix[r[b], r[b+1]]
                                new = distance_matrix[r[a-1], r[b]] + distance_matrix[r[a], r[b+1]]
                                if new < old - 1e-12:
                                    r[a:b+1] = reversed(r[a:b+1])
                                    route_dists[idx] = route_dist(r)
                                    improved = True
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
        return routes, route_dists, best_routes, best_max, best_total

    best_routes = None
    best_max = float('inf')
    best_total = float('inf')
    num_starts = min(10, n)
    for start in range(num_starts):
        routes, route_dists = construct_greedy()
        if best_routes is None:
            best_routes = [route[:] for route in routes]
            best_max = max(route_dists)
            best_total = total_dist(routes)
            report_best_vrp(best_routes)
        routes, route_dists = list(routes), list(route_dists)
        stagnation = 0
        for sub_iter in range(5):
            routes, route_dists, best_routes, best_max, best_total = improve(routes, route_dists, best_routes, best_max, best_total)
            cur_max = max(route_dists)
            cur_total = total_dist(routes)
            if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                best_max = cur_max
                best_total = cur_total
                best_routes = [route[:] for route in routes]
                report_best_vrp(best_routes)
                stagnation = 0
            else:
                stagnation += 1
            if stagnation >= 2:
                # perturbation: randomly remove up to 3 customers and reinsert greedily
                num_perturb = min(3, n-1)
                if num_perturb > 0:
                    customers = [c for c in range(1, n)]
                    random.shuffle(customers)
                    to_remove = customers[:num_perturb]
                    for c in to_remove:
                        for t in range(truck_count):
                            if c in routes[t]:
                                pos = routes[t].index(c)
                                pred = routes[t][pos-1]
                                succ = routes[t][pos+1]
                                route_dists[t] += distance_matrix[pred, succ] - distance_matrix[pred, c] - distance_matrix[c, succ]
                                routes[t].pop(pos)
                                break
                    unassigned = to_remove
                    random.shuffle(unassigned)
                    for c in unassigned:
                        best_new_max = float('inf')
                        best_truck = -1
                        best_pos = -1
                        for t in range(truck_count):
                            r = routes[t]
                            for pos in range(1, len(r)):
                                pred = r[pos-1]
                                succ = r[pos]
                                new_dist = route_dists[t] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                                other_max = 0.0
                                for k in range(truck_count):
                                    if k != t and route_dists[k] > other_max:
                                        other_max = route_dists[k]
                                overall_max = max(other_max, new_dist)
                                if overall_max < best_new_max - 1e-12:
                                    best_new_max = overall_max
                                    best_truck = t
                                    best_pos = pos
                        if best_truck != -1:
                            routes[best_truck].insert(best_pos, c)
                            route_dists[best_truck] = route_dist(routes[best_truck])
                stagnation = 0
    return best_routes