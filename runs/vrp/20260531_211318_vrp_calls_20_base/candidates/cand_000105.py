import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    def insertion_delta(route, pos, cust):
        prev = route[pos-1]
        nxt = route[pos]
        return dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]

    def decode(perm):
        # greedy insertion minimizing max distance
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        for cust in perm:
            best_truck = None
            best_pos = None
            best_max = float('inf')
            best_total = float('inf')
            for t, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_dist = route_dists[t] + insertion_delta(route, pos, cust)
                    new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                    new_total = sum(route_dists) + insertion_delta(route, pos, cust)
                    if new_max < best_max or (new_max == best_max and new_total < best_total):
                        best_max = new_max
                        best_total = new_total
                        best_truck = t
                        best_pos = pos
            route = routes[best_truck]
            routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
            route_dists[best_truck] += insertion_delta(route, best_pos, cust)
        return routes, route_dists

    def evaluate(routes, route_dists):
        current_max = max(route_dists)
        current_total = sum(route_dists)
        return current_max, current_total

    def two_opt(routes, route_dists, best_max, best_total):
        improved = True
        while improved:
            improved = False
            for t, route in enumerate(routes):
                if len(route) <= 3:
                    continue
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < route_dists[t] - 1e-9:
                            new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                            new_total = sum(route_dists[:t]) + new_dist + sum(route_dists[t+1:])
                            if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                                routes[t] = new_route
                                route_dists[t] = new_dist
                                best_max = new_max
                                best_total = new_total
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return routes, route_dists, best_max, best_total

    # Parameters
    pop_size = max(20, min(50, 10 * n))
    max_gen = max(50, min(100, 5 * n))
    crossover_rate = 0.9
    mutation_rate = 0.1
    tournament_size = 3
    elite_count = 2

    # Initial population
    customers = list(range(1, n))
    population = []
    for _ in range(pop_size):
        perm = list(customers)
        random.shuffle(perm)
        routes, route_dists = decode(perm)
        cmax, ctotal = evaluate(routes, route_dists)
        population.append((cmax, ctotal, perm, routes, route_dists))
    # sort by max, then total
    population.sort(key=lambda x: (x[0], x[1]))
    best_individual = population[0]
    report_best_vrp(best_individual[3])

    def crossover_ox(p1, p2):
        n_genes = len(p1)
        a = random.randint(0, n_genes-2)
        b = random.randint(a+1, n_genes-1)
        child1 = [None]*n_genes
        child2 = [None]*n_genes
        child1[a:b+1] = p1[a:b+1]
        child2[a:b+1] = p2[a:b+1]
        # fill remaining
        def fill_child(child, parent):
            pos = (b+1) % n_genes
            for gene in parent:
                if gene not in child:
                    child[pos] = gene
                    pos = (pos+1) % n_genes
        fill_child(child1, p2)
        fill_child(child2, p1)
        return child1, child2

    def mutate_swap(perm):
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
        return perm

    for gen in range(max_gen):
        new_population = []
        # Elitism
        for i in range(elite_count):
            new_population.append(population[i])
        # Generate rest
        while len(new_population) < pop_size:
            # tournament selection
            candidates = random.sample(range(pop_size), tournament_size)
            winner = min(candidates, key=lambda i: (population[i][0], population[i][1]))
            parent1 = population[winner][2]
            candidates = random.sample(range(pop_size), tournament_size)
            winner = min(candidates, key=lambda i: (population[i][0], population[i][1]))
            parent2 = population[winner][2]
            if random.random() < crossover_rate:
                child1_perm, child2_perm = crossover_ox(parent1, parent2)
            else:
                child1_perm, child2_perm = list(parent1), list(parent2)
            if random.random() < mutation_rate:
                child1_perm = mutate_swap(child1_perm)
            if random.random() < mutation_rate:
                child2_perm = mutate_swap(child2_perm)
            # decode
            routes1, dists1 = decode(child1_perm)
            cmax1, ctotal1 = evaluate(routes1, dists1)
            routes2, dists2 = decode(child2_perm)
            cmax2, ctotal2 = evaluate(routes2, dists2)
            new_population.append((cmax1, ctotal1, child1_perm, routes1, dists1))
            if len(new_population) < pop_size:
                new_population.append((cmax2, ctotal2, child2_perm, routes2, dists2))
        # keep only pop_size
        new_population.sort(key=lambda x: (x[0], x[1]))
        population = new_population[:pop_size]
        # update best
        if population[0][0] < best_individual[0] or (population[0][0] == best_individual[0] and population[0][1] < best_individual[1]):
            best_individual = population[0]
            report_best_vrp(best_individual[3])
    # post-optimization 2-opt on best
    routes, dists = best_individual[3], best_individual[4]
    best_max = best_individual[0]
    best_total = best_individual[1]
    routes, dists, best_max, best_total = two_opt(routes, dists, best_max, best_total)
    report_best_vrp(routes)
    return routes