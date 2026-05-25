import numpy as np
import math
import random
import heapq
import itertools
from collections import defaultdict

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(route, cust):
        best_pos = -1
        best_delta = float('inf')
        for pos in range(1, len(route)):
            prev = route[pos-1]
            nxt = route[pos]
            delta = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
        return best_pos, best_delta

    # 2-regret construction (deterministic)
    routes = [[depot, depot] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    while unassigned:
        cust_info = []
        for cust in unassigned:
            deltas = []
            positions = []
            for ridx, route in enumerate(routes):
                pos, delta = best_insertion(route, cust)
                deltas.append(delta)
                positions.append(pos)
            sorted_deltas = sorted(enumerate(deltas), key=lambda x: x[1])
            best_delta = sorted_deltas[0][1]
            second_best_delta = sorted_deltas[1][1] if len(sorted_deltas) > 1 else best_delta
            regret = second_best_delta - best_delta
            best_ridx = sorted_deltas[0][0]
            cust_info.append((-regret, best_delta, cust, best_ridx, positions[best_ridx]))
        # primary: highest regret (negative sign), tie-break: smallest best_delta, then cust
        cust_info.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, cust, ridx, pos = cust_info[0]
        routes[ridx].insert(pos, cust)
        unassigned.remove(cust)

    route_dists = [route_distance(r) for r in routes]
    best_routes = [r[:] for r in routes]
    best_max = max(route_dists)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    # Improvement phase
    max_iter = min(10, n)  # outer loops
    for _ in range(max_iter):
        improved_outer = False
        # Local search pass
        for _ in range(n * truck_count):  # inner bounded
            current_max = max(route_dists)
            max_idx = route_dists.index(current_max)
            improved = False
            route = routes[max_idx]
            # Inter-route relocate from max route
            for i in range(1, len(route)-1):
                cust = route[i]
                new_route = route[:i] + route[i+1:]
                new_dist = route_distance(new_route)
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for pos in range(1, len(other_route)):
                        new_other = other_route[:pos] + [cust] + other_route[pos:]
                        new_other_dist = route_distance(new_other)
                        candidate_max = max(new_dist, new_other_dist)
                        if candidate_max < current_max:
                            routes[max_idx] = new_route
                            routes[other_idx] = new_other
                            route_dists[max_idx] = new_dist
                            route_dists[other_idx] = new_other_dist
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                # Inter-route swap
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for i in range(1, len(route)-1):
                        for j in range(1, len(other_route)-1):
                            new_route = route[:i] + [other_route[j]] + route[i+1:]
                            new_other = other_route[:j] + [route[i]] + other_route[j+1:]
                            new_dist = route_distance(new_route)
                            new_other_dist = route_distance(new_other)
                            candidate_max = max(new_dist, new_other_dist)
                            if candidate_max < current_max:
                                routes[max_idx] = new_route
                                routes[other_idx] = new_other
                                route_dists[max_idx] = new_dist
                                route_dists[other_idx] = new_other_dist
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                # Intra-route 2-opt on max route
                best_route = route[:]
                best_dist = route_dists[max_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < best_dist:
                            best_route = new_route
                            best_dist = new_dist
                if best_dist < route_dists[max_idx]:
                    routes[max_idx] = best_route
                    route_dists[max_idx] = best_dist
                    improved = True

            if improved:
                improved_outer = True
                new_max = max(route_dists)
                if new_max < best_max:
                    best_max = new_max
                    best_routes = [r[:] for r in routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
            else:
                break  # no improvement in inner loop

        if not improved_outer:
            # Perturbation: move a random customer from longest to random other route at best insertion
            # but only if we have a best solution
            if best_routes is not None:
                perturb_routes = [r[:] for r in best_routes]
                perturb_dists = [route_distance(r) for r in perturb_routes]
                # find longest route in perturbed solution
                max_val = max(perturb_dists)
                long_idx = random.choice([i for i, d in enumerate(perturb_dists) if d == max_val])
                long_route = perturb_routes[long_idx]
                if len(long_route) > 2:
                    # pick random customer from long route
                    cust_pos = random.randint(1, len(long_route)-2)
                    cust = long_route[cust_pos]
                    new_long = long_route[:cust_pos] + long_route[cust_pos+1:]
                    # choose a different route at random
                    other_idx_list = [i for i in range(truck_count) if i != long_idx]
                    if other_idx_list:
                        other_idx = random.choice(other_idx_list)
                    else:
                        other_idx = (long_idx + 1) % truck_count
                    other_route = perturb_routes[other_idx]
                    pos, _ = best_insertion(other_route, cust)
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    # update routes for next outer iteration
                    routes = [r[:] for r in perturb_routes]
                    routes[long_idx] = new_long
                    routes[other_idx] = new_other
                    route_dists = [route_distance(r) for r in routes]
        # else continue with current routes

    while len(best_routes) < truck_count:
        best_routes.append([depot, depot])
    return best_routes