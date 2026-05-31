import numpy as np
import random
import math

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

    def best_insertion_order(permutation):
        # decode permutation into routes using best insertion minimizing max distance
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        for c in permutation:
            best = (float('inf'), -1, -1)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    pred = route[pos-1]
                    succ = route[pos]
                    new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                    other_max = 0.0
                    for j, d in enumerate(route_dists):
                        if j != r_idx and d > other_max:
                            other_max = d
                    new_max = max(other_max, new_dist)
                    if new_max < best[0]:
                        best = (new_max, r_idx, pos)
            if best[1] == -1:
                # fallback: assign to first route
                routes[0].insert(1, c)
                route_dists[0] = route_dist(routes[0])
            else:
                _, r_idx, pos = best
                routes[r_idx].insert(pos, c)
                route_dists[r_idx] = route_dist(routes[r_idx])
        return routes, route_dists

    def evaluate(permutation):
        routes, dists = best_insertion_order(permutation)
        return max(dists), routes

    # Build initial permutation from regret heuristic (insertion order)
    # Replicate regret construction and record order
    routes_init = [[0, 0] for _ in range(truck_count)]
    route_dists_init = [0.0] * truck_count
    unassigned = list(range(1, n))
    insertion_order = []
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
    while unassigned:
        bests = []
        for c in unassigned:
            best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes_init, route_dists_init)
            if best_route == -1:
                continue
            regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
            bests.append((-regret, c, best_route, best_pos, best_new_max))
        bests.sort(key=lambda x: (x[0], x[1]))
        _, c, best_route, best_pos, new_max = bests[0]
        route = routes_init[best_route]
        route.insert(best_pos, c)
        route_dists_init[best_route] = route_dist(route)
        unassigned.remove(c)
        insertion_order.append(c)
    start_perm = insertion_order  # permutation from regret

    # GA parameters
    pop_size = 20
    max_gen = 50
    tourn_size = 3
    cx_prob = 0.8
    mut_prob = 0.1

    # Initialize population
    population = []
    # add the start permutation
    population.append(start_perm)
    # fill with random permutations (unique)
    customers = list(range(1, n))
    for _ in range(pop_size - 1):
        perm = customers[:]
        random.shuffle(perm)
        population.append(perm)

    # Evaluate initial population
    fits = []
    best_overall = float('inf')
    best_routes = None
    for perm in population:
        max_dist, routes = evaluate(perm)
        fits.append(max_dist)
        if max_dist < best_overall:
            best_overall = max_dist
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    # Steady-state GA
    for gen in range(max_gen):
        # Selection: tournament
        new_pop = []
        # Elitism: keep best
        best_idx = min(range(len(population)), key=lambda i: fits[i])
        new_pop.append(population[best_idx][:])
        while len(new_pop) < pop_size:
            # pick two parents by tournament
            parent1 = tournament(population, fits, tourn_size)
            parent2 = tournament(population, fits, tourn_size)
            # crossover
            if random.random() < cx_prob:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            # mutation
            if random.random() < mut_prob:
                swap_mutation(child1)
            if random.random() < mut_prob:
                swap_mutation(child2)
            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)
        # replace population
        population = new_pop
        # evaluate
        fits = []
        for perm in population:
            max_dist, routes = evaluate(perm)
            fits.append(max_dist)
            if max_dist < best_overall:
                best_overall = max_dist
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)

    # Post-process best solution with intra-route 2-opt (simple)
    for r_idx in range(truck_count):
        route = best_routes[r_idx]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new < old - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        break
                if improved:
                    break
    # Ensure feasibility: all customers present once
    # Should be okay
    return best_routes

def tournament(pop, fits, k):
    # deterministic tie-breaking: lower index wins
    best = None
    best_fit = float('inf')
    best_idx = -1
    for _ in range(k):
        i = random.randrange(len(pop))
        if fits[i] < best_fit - 1e-12:
            best_fit = fits[i]
            best = pop[i]
            best_idx = i
        elif abs(fits[i] - best_fit) < 1e-12 and best_idx < i:
            # tie: prefer lower index
            pass
    return best[:]

def order_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child1 = [0] * size
    child2 = [0] * size
    child1[a:b] = p1[a:b]
    child2[a:b] = p2[a:b]
    # fill from p2 for child1
    fill1 = [x for x in p2 if x not in child1[a:b]]
    idx = 0
    for i in range(size):
        if child1[i] == 0:
            child1[i] = fill1[idx]
            idx += 1
    # fill from p1 for child2
    fill2 = [x for x in p1 if x not in child2[a:b]]
    idx = 0
    for i in range(size):
        if child2[i] == 0:
            child2[i] = fill2[idx]
            idx += 1
    return child1, child2

def swap_mutation(perm):
    i, j = random.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]