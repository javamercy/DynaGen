import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    # Helper functions
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
        routes = [[0, 0] for _ in range(truck_count)]
        dists = [0.0] * truck_count
        for cust in perm:
            best_truck = None
            best_pos = None
            best_max = float('inf')
            best_total = float('inf')
            best_delta = None
            for t in range(truck_count):
                route = routes[t]
                for pos in range(1, len(route)):
                    delta = insertion_delta(route, pos, cust)
                    new_dist = dists[t] + delta
                    new_max = max(dists[:t] + [new_dist] + dists[t+1:])
                    new_total = sum(dists) + delta
                    if new_max < best_max or (new_max == best_max and new_total < best_total):
                        best_max = new_max
                        best_total = new_total
                        best_truck = t
                        best_pos = pos
                        best_delta = delta
            route = routes[best_truck]
            routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
            dists[best_truck] += best_delta
        return routes, dists

    # Initial population
    pop_size = min(20, max(10, n // 5))
    max_gen = min(200, 10 * n)
    
    # Best ordering: farthest-first
    farthest_perm = sorted(range(1, n), key=lambda c: -dist[0][c])
    population = [farthest_perm[:]]
    while len(population) < pop_size:
        perm = list(range(1, n))
        random.shuffle(perm)
        population.append(perm)
    
    # Evaluate initial population
    def evaluate(perm):
        routes, dists = decode(perm)
        return max(dists), sum(dists), routes
    
    fitness = [evaluate(perm) for perm in population]
    best_idx = min(range(len(fitness)), key=lambda i: (fitness[i][0], fitness[i][1]))
    best_max, best_total, best_routes = fitness[best_idx]
    report_best_vrp(best_routes)

    # Genetic algorithm
    for gen in range(max_gen):
        new_population = []
        # Elitism: keep best individual
        new_population.append(population[best_idx])
        
        while len(new_population) < pop_size:
            # Tournament selection
            tournament = random.sample(range(pop_size), 3)
            parent1_idx = min(tournament, key=lambda i: (fitness[i][0], fitness[i][1]))
            tournament = random.sample(range(pop_size), 3)
            parent2_idx = min(tournament, key=lambda i: (fitness[i][0], fitness[i][1]))
            p1 = population[parent1_idx]
            p2 = population[parent2_idx]
            
            # Crossover (PMX)
            if random.random() < 0.8:
                size = len(p1)
                cx1 = random.randint(1, size-2)
                cx2 = random.randint(cx1+1, size-1)
                child1 = [None]*size
                child2 = [None]*size
                # copy middle segment
                child1[cx1:cx2+1] = p1[cx1:cx2+1]
                child2[cx1:cx2+1] = p2[cx1:cx2+1]
                # fill remaining from other parent
                def fill_child(child, other, start, end):
                    pos = (end + 1) % size
                    for i in range(size):
                        c = other[(start + i) % size]
                        if c not in child:
                            child[pos] = c
                            pos = (pos + 1) % size
                    return child
                child1 = fill_child(child1, p2, cx2+1, cx1-1)
                child2 = fill_child(child2, p1, cx2+1, cx1-1)
                child1 = fill_child(child1, p2, 0, cx1-1)  # ensure all filled
                child2 = fill_child(child2, p1, 0, cx1-1)
                # fill remaining None (should not happen)
                for i in range(size):
                    if child1[i] is None:
                        for c in p2:
                            if c not in child1:
                                child1[i] = c
                                break
                    if child2[i] is None:
                        for c in p1:
                            if c not in child2:
                                child2[i] = c
                                break
            else:
                child1 = p1[:]
                child2 = p2[:]
            
            # Mutation: swap two customers
            if random.random() < 0.1:
                i, j = random.sample(range(len(child1)), 2)
                child1[i], child1[j] = child1[j], child1[i]
            if random.random() < 0.1:
                i, j = random.sample(range(len(child2)), 2)
                child2[i], child2[j] = child2[j], child2[i]
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        # Replace population
        population = new_population[:pop_size]
        fitness = [evaluate(perm) for perm in population]
        best_idx = min(range(len(fitness)), key=lambda i: (fitness[i][0], fitness[i][1]))
        if fitness[best_idx][0] < best_max - 1e-9 or (abs(fitness[best_idx][0] - best_max) < 1e-9 and fitness[best_idx][1] < best_total):
            best_max, best_total, best_routes = fitness[best_idx]
            report_best_vrp(best_routes)
    
    return best_routes