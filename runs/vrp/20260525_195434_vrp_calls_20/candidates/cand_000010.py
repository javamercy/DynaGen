import numpy as np
import random
import math
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

    def construct_routes():
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
                top_deltas = [d for _, d in sorted_deltas[:3]]
                if len(top_deltas) < 3:
                    regret = sum(top_deltas) - top_deltas[0]
                else:
                    regret = top_deltas[2] - top_deltas[0]
                best_ridx = sorted_deltas[0][0]
                best_delta = sorted_deltas[0][1]
                cust_info.append((regret, best_delta, cust, best_ridx, positions[best_ridx]))
            # Stochastic selection: with 0.5 probability choose best regret, else random among top 3 by regret
            if random.random() < 0.5:
                cust_info.sort(key=lambda x: (-x[0], x[1], x[2]))
                _, _, cust, ridx, pos = cust_info[0]
            else:
                top3 = sorted(cust_info, key=lambda x: -x[0])[:3]
                choice = random.choice(top3)
                _, _, cust, ridx, pos = choice
            routes[ridx].insert(pos, cust)
            unassigned.remove(cust)
        return routes

    best_routes = None
    best_max = float('inf')
    max_iter = min(5, n)
    for _ in range(max_iter):
        routes = construct_routes()
        route_dists = [route_distance(r) for r in routes]
        current_max = max(route_dists)
        # Local search (steepest descent)
        improved = True
        while improved:
            improved = False
            for max_idx in [i for i, d in enumerate(route_dists) if d == current_max]:
                route = routes[max_idx]
                found = False
                # Inter-route relocate
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
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    continue
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
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    continue
                # Intra-route 2-opt
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
                current_max = max(route_dists)
        # After local search, check if best
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            try:
                report_best_vrp(best_routes)
            except NameError:
                pass
        # Perturbation: random inter-route swap perturbations
        if _ < max_iter - 1 and best_routes is not None:
            perturb_routes = [r[:] for r in best_routes]
            num_perturb = random.randint(1, min(3, truck_count))
            for _ in range(num_perturb):
                route_indices = [i for i, r in enumerate(perturb_routes) if len(r) > 2]
                if len(route_indices) < 2:
                    break
                idx1, idx2 = random.sample(route_indices, 2)
                route1 = perturb_routes[idx1]
                route2 = perturb_routes[idx2]
                pos1 = random.randint(1, len(route1)-2)
                pos2 = random.randint(1, len(route2)-2)
                cust1 = route1[pos1]
                cust2 = route2[pos2]
                route1[pos1] = cust2
                route2[pos2] = cust1
                perturb_routes[idx1] = route1
                perturb_routes[idx2] = route2
            routes = perturb_routes
            route_dists = [route_distance(r) for r in perturb_routes]
            current_max = max(route_dists)
    while len(best_routes) < truck_count:
        best_routes.append([depot, depot])
    return best_routes