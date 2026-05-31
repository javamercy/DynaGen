import numpy as np
import random
import math

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

    def regret_construction(routes, route_dists, unassigned):
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
        return routes, route_dists

    def improve(routes, route_dists):
        # Intra-route 2-opt
        for r_idx in range(truck_count):
            improved = True
            while improved:
                improved = False
                route = routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                        new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                        if new < old - 1e-12:
                            route[i:j+1] = reversed(route[i:j+1])
                            improved = True
                            route_dists[r_idx] = route_dist(route)
                            break
                    if improved:
                        break
        # Inter-route improvement loop
        max_iter = n * truck_count
        for _ in range(max_iter):
            improved_overall = False
            # Best-improvement relocate from longest route
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
                for r_idx in [max_idx, other_idx]:
                    improved = True
                    while improved:
                        improved = False
                        route = routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved = True
                                    route_dists[r_idx] = route_dist(route)
                                    break
                            if improved:
                                break
                improved_overall = True

            if not improved_overall:
                # Best swap
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
                    route_max = routes[max_idx]
                    other_route = routes[other_idx]
                    c1 = route_max[i]
                    c2 = other_route[j]
                    route_max[i] = c2
                    other_route[j] = c1
                    route_dists[max_idx] = new_dist_max
                    route_dists[other_idx] = new_dist_other
                    for r_idx in [max_idx, other_idx]:
                        improved = True
                        while improved:
                            improved = False
                            route = routes[r_idx]
                            for a in range(1, len(route)-2):
                                for b in range(a+1, len(route)-1):
                                    old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                    new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                    if new < old - 1e-12:
                                        route[a:b+1] = reversed(route[a:b+1])
                                        improved = True
                                        route_dists[r_idx] = route_dist(route)
                                        break
                                if improved:
                                    break
                    improved_overall = True

            if not improved_overall:
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
                    route_max = routes[max_idx]
                    other_route = routes[other_idx]
                    new_route_max = route_max[:i+1] + other_route[j+1:]
                    new_route_other = other_route[:j+1] + route_max[i+1:]
                    routes[max_idx] = new_route_max
                    routes[other_idx] = new_route_other
                    route_dists[max_idx] = route_dist(new_route_max)
                    route_dists[other_idx] = route_dist(new_route_other)
                    for r_idx in [max_idx, other_idx]:
                        improved = True
                        while improved:
                            improved = False
                            route = routes[r_idx]
                            for a in range(1, len(route)-2):
                                for b in range(a+1, len(route)-1):
                                    old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                    new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                    if new < old - 1e-12:
                                        route[a:b+1] = reversed(route[a:b+1])
                                        improved = True
                                        route_dists[r_idx] = route_dist(route)
                                        break
                                if improved:
                                    break
                    improved_overall = True

            if not improved_overall:
                break
        return routes, route_dists

    # Initial population
    pop_size = min(10, n)
    population = []
    # First individual: regret greedy
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))
    routes, route_dists = regret_construction(routes, route_dists, unassigned)
    routes, route_dists = improve(routes, route_dists)
    population.append(([r[:] for r in routes], route_dists[:], max(route_dists), total_dist(routes)))
    report_best_vrp(routes)
    # Generate more individuals by perturbing initial solution
    for _ in range(pop_size - 1):
        routes_copy = [r[:] for r in routes]
        route_dists_copy = route_dists[:]
        # Random perturbation: move some customers
        for __ in range(random.randint(1, max(1, n//10))):
            max_idx = route_dists_copy.index(max(route_dists_copy))
            r = random.randint(0, truck_count-1)
            if len(routes_copy[max_idx]) > 2:
                pos = random.randint(1, len(routes_copy[max_idx])-2)
                c = routes_copy[max_idx].pop(pos)
                new_route_idx = random.randint(0, truck_count-1)
                new_pos = random.randint(1, len(routes_copy[new_route_idx])-1)
                routes_copy[new_route_idx].insert(new_pos, c)
                route_dists_copy[max_idx] = route_dist(routes_copy[max_idx])
                route_dists_copy[new_route_idx] = route_dist(routes_copy[new_route_idx])
        routes_copy, route_dists_copy = improve(routes_copy, route_dists_copy)
        max_dist = max(route_dists_copy)
        total = total_dist(routes_copy)
        population.append(([r[:] for r in routes_copy], route_dists_copy[:], max_dist, total))
        report_best_vrp(routes_copy)

    # Sort population by max dist, then total
    population.sort(key=lambda x: (x[2], x[3]))
    best_solution = ([r[:] for r in population[0][0]], population[0][2])

    # Genetic algorithm
    max_generations = min(20, n * 2)
    for gen in range(max_generations):
        # selection: binary tournament
        def tournament_select():
            i1 = random.randint(0, pop_size-1)
            i2 = random.randint(0, pop_size-1)
            if population[i1][2] < population[i2][2] - 1e-12:
                return i1
            elif population[i2][2] < population[i1][2] - 1e-12:
                return i2
            else:
                return i1 if population[i1][3] < population[i2][3] else i2

        p1_idx = tournament_select()
        p2_idx = tournament_select()
        parent1 = population[p1_idx][0]
        parent2 = population[p2_idx][0]

        # Crossover: route exchange
        num_routes_copy = random.randint(1, truck_count - 1)
        route_indices = random.sample(range(truck_count), num_routes_copy)
        child_routes = [None] * truck_count
        used_customers = set()
        for idx in route_indices:
            child_routes[idx] = parent1[idx][:]
            for node in parent1[idx]:
                if node != 0:
                    used_customers.add(node)
        # Fill remaining routes with empty
        remaining_indices = [i for i in range(truck_count) if i not in route_indices]
        for idx in remaining_indices:
            child_routes[idx] = [0, 0]
        # Collect unassigned customers
        unassigned = [c for c in range(1, n) if c not in used_customers]
        # Use parent2 order for insertion (but simplistic: use regret on empty routes)
        # We'll use regret insertion for remaining routes
        child_route_dists = [route_dist(r) for r in child_routes]
        child_routes, child_route_dists = regret_construction(child_routes, child_route_dists, unassigned)
        # Mutation: move random customer
        if random.random() < 0.2:
            # choose random customer from a random route
            max_attempts = 10
            for _ in range(max_attempts):
                src_idx = random.randint(0, truck_count-1)
                if len(child_routes[src_idx]) > 2:
                    break
            else:
                src_idx = random.randint(0, truck_count-1)
            if len(child_routes[src_idx]) > 2:
                pos = random.randint(1, len(child_routes[src_idx])-2)
                c = child_routes[src_idx].pop(pos)
                dst_idx = random.randint(0, truck_count-1)
                new_pos = random.randint(1, len(child_routes[dst_idx])-1)
                child_routes[dst_idx].insert(new_pos, c)
                child_route_dists[src_idx] = route_dist(child_routes[src_idx])
                child_route_dists[dst_idx] = route_dist(child_routes[dst_idx])
        # Improve offspring
        child_routes, child_route_dists = improve(child_routes, child_route_dists)
        child_max = max(child_route_dists)
        child_total = total_dist(child_routes)
        if child_max < best_solution[1] - 1e-12:
            best_solution = ([r[:] for r in child_routes], child_max)
            report_best_vrp(child_routes)
        # Replace worst individual if offspring is better
        worst_idx = pop_size - 1
        if child_max < population[worst_idx][2] - 1e-12 or (abs(child_max - population[worst_idx][2]) < 1e-12 and child_total < population[worst_idx][3]):
            population[worst_idx] = ([r[:] for r in child_routes], child_route_dists[:], child_max, child_total)
            population.sort(key=lambda x: (x[2], x[3]))
            if population[0][2] < best_solution[1] - 1e-12:
                best_solution = ([r[:] for r in population[0][0]], population[0][2])
                report_best_vrp(population[0][0])

    return best_solution[0]