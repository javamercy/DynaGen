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

    def probabilistic_regret_insertion(unassigned, routes):
        # probabilistic 2-regret: compute regret for each customer, then select based on softmax
        cust_weights = []
        for cust in unassigned:
            deltas = []
            positions = []
            for ridx, route in enumerate(routes):
                if len(route) == 2: # allow empty? but routes always have depot
                    pass
                pos, delta = best_insertion(route, cust)
                deltas.append(delta)
                positions.append(pos)
            sorted_deltas = sorted(enumerate(deltas), key=lambda x: x[1])
            best_delta = sorted_deltas[0][1]
            second_best_delta = sorted_deltas[1][1] if len(sorted_deltas) > 1 else best_delta
            regret = second_best_delta - best_delta
            # weight: exp(regret) to avoid zero weight
            weight = math.exp(min(regret, 100))  # cap to avoid overflow
            cust_weights.append((weight, cust, best_delta, deltas, positions))
        # normalize weights and pick randomly
        total_weight = sum(w for w,_,_,_,_ in cust_weights)
        if total_weight == 0:
            # uniform
            idx = random.randrange(len(cust_weights))
        else:
            r = random.random() * total_weight
            cumul = 0.0
            idx = 0
            for i, (w,_,_,_,_) in enumerate(cust_weights):
                cumul += w
                if r <= cumul:
                    idx = i
                    break
        _, cust, best_delta, deltas, positions = cust_weights[idx]
        # find route that gave best_delta
        best_ridx = deltas.index(min(deltas))
        return cust, best_ridx, positions[best_ridx]

    def construction():
        routes = [[depot, depot] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        random.shuffle(unassigned)  # random order for diversity
        while unassigned:
            cust, ridx, pos = probabilistic_regret_insertion(unassigned, routes)
            routes[ridx].insert(pos, cust)
            unassigned.remove(cust)
        return routes

    def destroy_recreate(routes):
        # remove 10-20% of customers randomly
        num_remove = max(1, int(n * random.uniform(0.1, 0.2)))
        all_custs = list(range(1, n))
        random.shuffle(all_custs)
        to_remove = all_custs[:num_remove]
        # remove from routes
        for ridx, route in enumerate(routes):
            routes[ridx] = [c for c in route if c not in to_remove]
            # ensure start/end depot
            if routes[ridx][0] != depot:
                routes[ridx].insert(0, depot)
            if routes[ridx][-1] != depot:
                routes[ridx].append(depot)
        # reinsert using probabilistic regret
        unassigned = to_remove
        while unassigned:
            cust, ridx, pos = probabilistic_regret_insertion(unassigned, routes)
            routes[ridx].insert(pos, cust)
            unassigned.remove(cust)
        return routes

    best_routes = None
    best_max = float('inf')

    for restart in range(3):  # multiple restarts
        routes = construction()
        route_dists = [route_distance(r) for r in routes]
        current_routes = [r[:] for r in routes]
        current_dists = route_dists[:]
        current_max = max(current_dists)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in current_routes]
            try:
                report_best_vrp(best_routes)
            except NameError:
                pass

        # improvement loops
        max_iter = min(10, n)  # outer passes
        for _ in range(max_iter):
            improved_outer = False
            # inner local search pass
            for _ in range(n * truck_count * 2):
                improved = False
                # randomly select a route to improve (weighted by length? just uniform)
                ridx = random.randrange(truck_count)
                route = current_routes[ridx]
                current_rid_dist = current_dists[ridx]
                # choose a random move type
                move_type = random.choice(['relocate', 'swap', 'two_opt', 'three_opt'])
                if move_type == 'relocate':
                    # inter-route relocate: move a customer from this route to another
                    if len(route) <= 2:
                        continue
                    i = random.randint(1, len(route)-2)
                    cust = route[i]
                    new_route = route[:i] + route[i+1:]
                    new_dist = route_distance(new_route)
                    other_idx = random.choice([j for j in range(truck_count) if j != ridx])
                    other_route = current_routes[other_idx]
                    pos, delta = best_insertion(other_route, cust)
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_other_dist = route_distance(new_other)
                    candidate_max = max(new_dist, new_other_dist, current_max if 0 else 0)
                    # compute new overall max
                    new_max = max(new_dist, new_other_dist)
                    for k, dist in enumerate(current_dists):
                        if k not in [ridx, other_idx]:
                            new_max = max(new_max, dist)
                    if new_max < current_max:
                        current_routes[ridx] = new_route
                        current_routes[other_idx] = new_other
                        current_dists[ridx] = new_dist
                        current_dists[other_idx] = new_other_dist
                        current_max = new_max
                        improved = True
                elif move_type == 'swap':
                    # inter-route swap between two random routes
                    other_idx = random.choice([j for j in range(truck_count) if j != ridx])
                    other_route = current_routes[other_idx]
                    if len(route) <= 2 or len(other_route) <= 2:
                        continue
                    i = random.randint(1, len(route)-2)
                    j = random.randint(1, len(other_route)-2)
                    new_route = route[:i] + [other_route[j]] + route[i+1:]
                    new_other = other_route[:j] + [route[i]] + other_route[j+1:]
                    new_dist = route_distance(new_route)
                    new_other_dist = route_distance(new_other)
                    new_max = max(new_dist, new_other_dist)
                    for k, dist in enumerate(current_dists):
                        if k not in [ridx, other_idx]:
                            new_max = max(new_max, dist)
                    if new_max < current_max:
                        current_routes[ridx] = new_route
                        current_routes[other_idx] = new_other
                        current_dists[ridx] = new_dist
                        current_dists[other_idx] = new_other_dist
                        current_max = new_max
                        improved = True
                elif move_type == 'two_opt':
                    # intra-route 2-opt on selected route
                    best_route = route[:]
                    best_dist = current_rid_dist
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_dist = route_distance(new_route)
                            if new_dist < best_dist:
                                best_route = new_route
                                best_dist = new_dist
                    if best_dist < current_rid_dist:
                        current_routes[ridx] = best_route
                        current_dists[ridx] = best_dist
                        new_max = max(current_dists)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                elif move_type == 'three_opt':
                    # intra-route 3-opt on selected route (random try)
                    if len(route) < 6:
                        continue
                    # just try a few random 3-opt moves
                    for _ in range(10):  # bounded
                        i = random.randint(1, len(route)-4)
                        j = random.randint(i+2, len(route)-3)
                        k = random.randint(j+2, len(route)-2)
                        # several reversal patterns
                        option = random.choice(['reverse1', 'reverse2', 'reverse3'])
                        if option == 'reverse1':
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        elif option == 'reverse2':
                            new_route = route[:i] + route[i:j+1] + route[j+1:k+1][::-1] + route[k+1:]
                        else:
                            new_route = route[:i] + route[i:j+1] + route[j+1:k+1][::-1] + route[k+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < current_rid_dist:
                            current_routes[ridx] = new_route
                            current_dists[ridx] = new_dist
                            new_max = max(current_dists)
                            if new_max < current_max:
                                current_max = new_max
                                improved = True
                                break
                if improved:
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [r[:] for r in current_routes]
                        try:
                            report_best_vrp(best_routes)
                        except NameError:
                            pass
                    improved_outer = True
                    break  # break inner loop after improvement
            if not improved_outer:
                # destroy and recreate perturbation
                current_routes = destroy_recreate(current_routes)
                current_dists = [route_distance(r) for r in current_routes]
                current_max = max(current_dists)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = [r[:] for r in current_routes]
                    try:
                        report_best_vrp(best_routes)
                    except NameError:
                        pass
    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([depot, depot])
    return best_routes