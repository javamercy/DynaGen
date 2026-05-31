import numpy as np
import random
from itertools import accumulate

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count <= 0:
        return []
    customers = list(range(1, n))
    INF = 1e15

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def two_opt(route):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < route_dist(route) - 1e-12:
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
        return route

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

    def regret_construction(rand_tie=False):
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unassigned = list(range(1, n))
        while unassigned:
            candidates = []
            for c in unassigned:
                best_val, best_route, best_pos, second_val = best_insertion(c, routes, route_dists)
                if best_route == -1:
                    continue
                regret = second_val - best_val if second_val != INF else INF
                candidates.append((-regret, c, best_route, best_pos, best_val))
            if not candidates:
                break
            max_regret = max(x[0] for x in candidates)
            top = [x for x in candidates if abs(x[0] - max_regret) < 1e-12]
            if rand_tie and len(top) > 1:
                chosen = random.choice(top)
            else:
                top.sort(key=lambda x: x[1])
                chosen = top[0]
            _, c, best_route, best_pos, _ = chosen
            route = routes[best_route]
            route.insert(best_pos, c)
            route_dists[best_route] = route_dist(route)
            unassigned.remove(c)
        # Apply 2-opt to all routes
        for r_idx in range(truck_count):
            routes[r_idx] = two_opt(routes[r_idx])
            route_dists[r_idx] = route_dist(routes[r_idx])
        return routes, route_dists

    # Generate initial population
    pop_size = min(20, max(10, n // 5))
    population = []
    # First solution using deterministic regret
    routes, dists = regret_construction(rand_tie=False)
    population.append((routes, dists))
    for _ in range(pop_size - 1):
        routes, dists = regret_construction(rand_tie=True)
        population.append((routes, dists))

    def evaluate(routes):
        return max(route_dist(r) for r in routes)

    best_routes = None
    best_max = INF
    for routes, _ in population:
        m = evaluate(routes)
        if m < best_max - 1e-12:
            best_max = m
            best_routes = [route[:] for route in routes]
    report_best_vrp(best_routes)

    # Local search procedure (from parents)
    def local_search(routes):
        route_dists = [route_dist(r) for r in routes]
        improved = True
        max_iter = n * truck_count
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # Relocate from longest route
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
                # 2-opt on affected routes
                routes[max_idx] = two_opt(routes[max_idx])
                routes[other_idx] = two_opt(routes[other_idx])
                route_dists[max_idx] = route_dist(routes[max_idx])
                route_dists[other_idx] = route_dist(routes[other_idx])
                improved = True
                continue
            # Swap from longest route
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
                        old1 = route_dists[max_idx]
                        old2 = route_dists[other_idx]
                        pred1 = route_max[i-1]
                        succ1 = route_max[i+1]
                        new_dist_max = old1 - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                        pred2 = other_route[j-1]
                        succ2 = other_route[j+1]
                        new_dist_other = old2 - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
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
                route_max = routes[max_idx]
                other_route = routes[other_idx]
                c1 = route_max[i]
                c2 = other_route[j]
                route_max[i] = c2
                other_route[j] = c1
                route_dists[max_idx] = new_dist_max
                route_dists[other_idx] = new_dist_other
                routes[max_idx] = two_opt(routes[max_idx])
                routes[other_idx] = two_opt(routes[other_idx])
                route_dists[max_idx] = route_dist(routes[max_idx])
                route_dists[other_idx] = route_dist(routes[other_idx])
                improved = True
                continue
            # 2-opt* from longest route
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
                        if route_max[-1] != 0 or other_route[-1] != 0:
                            continue
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
                routes[max_idx] = two_opt(routes[max_idx])
                routes[other_idx] = two_opt(routes[other_idx])
                route_dists[max_idx] = route_dist(routes[max_idx])
                route_dists[other_idx] = route_dist(routes[other_idx])
                improved = True
        return routes, route_dists

    # Genetic Algorithm
    max_gen = min(10, n // 10)
    for gen in range(max_gen):
        new_pop = []
        # Elitism: keep best solution
        fitness = [evaluate(routes) for routes, _ in population]
        best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
        new_pop.append((population[best_idx][0], population[best_idx][1]))
        while len(new_pop) < pop_size:
            # Tournament selection
            tourn_size = 3
            idx1 = random.randint(0, pop_size-1)
            for _ in range(tourn_size-1):
                idx = random.randint(0, pop_size-1)
                if fitness[idx] < fitness[idx1]:
                    idx1 = idx
            idx2 = random.randint(0, pop_size-1)
            for _ in range(tourn_size-1):
                idx = random.randint(0, pop_size-1)
                if fitness[idx] < fitness[idx2]:
                    idx2 = idx
            parent1 = population[idx1][0]
            parent2 = population[idx2][0]
            # Convert routes to giant tours (permutation of customers)
            gt1 = []
            for r in parent1:
                for node in r:
                    if node != 0:
                        gt1.append(node)
            gt2 = []
            for r in parent2:
                for node in r:
                    if node != 0:
                        gt2.append(node)
            # Order Crossover (OX)
            size = len(gt1)
            if size < 2:
                offspring_gt = gt1[:]
            else:
                a = random.randint(0, size-1)
                b = random.randint(0, size-1)
                if a > b:
                    a, b = b, a
                offspring_gt = [None] * size
                offspring_gt[a:b+1] = gt1[a:b+1]
                pointer = (b+1) % size
                for i in range(size):
                    idx = (b+1 + i) % size
                    if gt2[idx] not in offspring_gt:
                        offspring_gt[pointer] = gt2[idx]
                        pointer = (pointer + 1) % size
                # Fill any remaining slots
                for i in range(size):
                    if offspring_gt[i] is None:
                        for val in gt2:
                            if val not in offspring_gt:
                                offspring_gt[i] = val
                                break
            # Mutation: swap two customers
            if random.random() < 0.2 and size >= 2:
                i = random.randint(0, size-1)
                j = random.randint(0, size-1)
                offspring_gt[i], offspring_gt[j] = offspring_gt[j], offspring_gt[i]
            # Convert giant tour to routes using simple split (nearest insertion? but we have regret construction)
            # Instead, we create routes by inserting in order but we need to respect truck count. Use nearest insertion heuristic similar to construction?
            # For simplicity, we use the regret construction on the giant tour order: we treat the order as a seed for insertion?
            # Better: we can use the giant tour as a guideline and then apply a fast insertion heuristic that respects the order.
            # We'll use a simple procedure: start with empty routes, then insert customers one by one in order of giant tour, using best insertion with tie-breaking.
            # This is similar to the construction but with fixed order.
            routes_new = [[0, 0] for _ in range(truck_count)]
            route_dists_new = [0.0] * truck_count
            for c in offspring_gt:
                _, best_route, best_pos, _ = best_insertion(c, routes_new, route_dists_new)
                if best_route == -1:
                    best_route = 0
                    best_pos = 1
                routes_new[best_route].insert(best_pos, c)
                route_dists_new[best_route] = route_dist(routes_new[best_route])
            # Apply 2-opt
            for r_idx in range(truck_count):
                routes_new[r_idx] = two_opt(routes_new[r_idx])
                route_dists_new[r_idx] = route_dist(routes_new[r_idx])
            # Add to new population
            new_pop.append((routes_new, route_dists_new))
        # Replace population
        population = new_pop
        # Evaluate and report best
        for routes, _ in population:
            m = evaluate(routes)
            if m < best_max - 1e-12:
                best_max = m
                best_routes = [route[:] for route in routes]
                report_best_vrp(best_routes)
    # Final local search on best
    routes, _ = local_search([route[:] for route in best_routes])
    m = evaluate(routes)
    if m < best_max - 1e-12:
        best_max = m
        best_routes = [route[:] for route in routes]
        report_best_vrp(best_routes)
    return best_routes