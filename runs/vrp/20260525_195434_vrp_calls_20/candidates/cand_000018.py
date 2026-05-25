import numpy as np
import random
import math
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
                    regret = sum(top_deltas) - top_deltas[0] if top_deltas else 0.0
                else:
                    regret = top_deltas[2] - top_deltas[0]
                best_ridx = sorted_deltas[0][0]
                best_delta = sorted_deltas[0][1]
                cust_info.append((regret, best_delta, cust, best_ridx, positions[best_ridx]))
            # Stochastic selection
            if not cust_info:
                break
            if random.random() < 0.5:
                cust_info.sort(key=lambda x: (-x[0], x[1], x[2]))
                _, _, cust, ridx, pos = cust_info[0]
            else:
                # choose among top 3 by regret
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
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            try:
                report_best_vrp(best_routes)
            except NameError:
                pass
        # Perturbation
        if _ < max_iter - 1 and best_routes is not None and truck_count > 1:
            perturb_routes = [r[:] for r in best_routes]
            perturb_dists = [route_distance(r) for r in perturb_routes]
            max_val = max(perturb_dists)
            long_indices = [i for i, d in enumerate(perturb_dists) if d == max_val]
            num_perturb = random.randint(1, min(3, n-1))
            for _ in range(num_perturb):
                if not long_indices:
                    break
                src_idx = random.choice(long_indices)
                src_route = perturb_routes[src_idx]
                if len(src_route) <= 2:
                    break
                cust_pos = random.randint(1, len(src_route)-2)
                cust = src_route[cust_pos]
                new_src = src_route[:cust_pos] + src_route[cust_pos+1:]
                dest_idx = random.randrange(truck_count)
                while dest_idx == src_idx:
                    dest_idx = random.randrange(truck_count)
                dest_route = perturb_routes[dest_idx]
                insert_pos = random.randint(1, len(dest_route)-1)
                new_dest = dest_route[:insert_pos] + [cust] + dest_route[insert_pos:]
                perturb_routes[src_idx] = new_src
                perturb_routes[dest_idx] = new_dest
                perturb_dists = [route_distance(r) for r in perturb_routes]
                max_val = max(perturb_dists)
                long_indices = [i for i, d in enumerate(perturb_dists) if d == max_val]
            routes = perturb_routes
            route_dists = perturb_dists
            current_max = max(route_dists)
    # Ensure exactly truck_count routes and validity
    if best_routes is None:
        best_routes = [[depot, depot] for _ in range(truck_count)]
    while len(best_routes) < truck_count:
        best_routes.append([depot, depot])
    # Validate: check all customers covered exactly once
    all_nodes = []
    for r in best_routes:
        all_nodes.extend(r[1:-1])
    if set(all_nodes) != set(range(1, n)) or len(all_nodes) != n-1:
        # fallback: construct simple solution
        best_routes = [[depot, depot] for _ in range(truck_count)]
        customers = list(range(1, n))
        for i, cust in enumerate(customers):
            best_routes[i % truck_count].insert(1, cust)
    return best_routes