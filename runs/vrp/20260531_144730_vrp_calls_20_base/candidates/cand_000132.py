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

    def total_dist(routes):
        return sum(route_dist(r) for r in routes)

    def best_insertion(c, routes, route_dists):
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            other_max = max(route_dists[j] for j in range(truck_count) if j != r_idx) if truck_count > 1 else 0.0
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

    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        for c in perm:
            best_max, best_route, best_pos, _ = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            routes[best_route].insert(best_pos, c)
            route_dists[best_route] += distance_matrix[routes[best_route][best_pos-1], c] + distance_matrix[c, routes[best_route][best_pos+1]] - distance_matrix[routes[best_route][best_pos-1], routes[best_route][best_pos+1]]
        # normalize route_dists
        for i in range(truck_count):
            route_dists[i] = route_dist(routes[i])
        return routes, route_dists

    def improve(routes, route_dists):
        # Intra-route 2-opt on all routes
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
                            break
                    if improved:
                        break
                if improved:
                    route_dists[r_idx] = route_dist(route)
        return routes, route_dists

    def crossover(p1, p2):
        # Order crossover (OX)
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        remaining = [x for x in p2 if x not in child[a:b+1]]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[idx]
                idx += 1
        return child

    def mutate(perm):
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
        return perm

    # Initial population
    pop_size = min(20, n)
    population = []
    fitness = []
    for _ in range(pop_size):
        perm = list(range(1, n))
        random.shuffle(perm)
        routes, route_dists = decode(perm)
        routes, route_dists = improve(routes, route_dists)
        max_dist = max(route_dists)
        fitness.append(max_dist)
        population.append((perm, routes, route_dists))
    best_idx = min(range(pop_size), key=lambda i: fitness[i])
    best_routes = [route[:] for route in population[best_idx][1]]
    best_max = fitness[best_idx]
    report_best_vrp(best_routes)

    generations = min(20, n * 2)
    for gen in range(generations):
        new_pop = []
        new_fit = []
        # Elitism: keep best solution
        new_pop.append(population[best_idx])
        new_fit.append(fitness[best_idx])
        while len(new_pop) < pop_size:
            # Tournament selection
            i1, i2 = random.sample(range(len(population)), 2)
            parent1 = population[i1] if fitness[i1] < fitness[i2] else population[i2]
            i1, i2 = random.sample(range(len(population)), 2)
            parent2 = population[i1] if fitness[i1] < fitness[i2] else population[i2]
            # Crossover
            if random.random() < 0.8:
                child_perm = crossover(parent1[0], parent2[0])
            else:
                child_perm = parent1[0][:]
            # Mutation
            if random.random() < 0.1:
                child_perm = mutate(child_perm)
            # Decode and improve
            child_routes, child_dists = decode(child_perm)
            child_routes, child_dists = improve(child_routes, child_dists)
            child_fit = max(child_dists)
            new_pop.append((child_perm, child_routes, child_dists))
            new_fit.append(child_fit)
        population = new_pop
        fitness = new_fit
        # Update best
        for i in range(len(population)):
            if fitness[i] < best_max - 1e-12:
                best_max = fitness[i]
                best_routes = [route[:] for route in population[i][1]]
                report_best_vrp(best_routes)
    return best_routes