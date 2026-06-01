import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    def route_dist(route):
        total = 0.0
        for a in range(len(route) - 1):
            total += distance_matrix[route[a], route[a+1]]
        return total

    def tour_to_routes(tour):
        seg_dist = [[0.0] * (m + 1) for _ in range(m)]
        for l in range(m):
            acc = distance_matrix[0, tour[l]]
            for r in range(l + 1, m + 1):
                if r > l + 1:
                    acc += distance_matrix[tour[r-2], tour[r-1]]
                if r == l + 1:
                    route_d = distance_matrix[0, tour[l]] + distance_matrix[tour[l], 0]
                else:
                    route_d = acc + distance_matrix[tour[r-1], 0]
                seg_dist[l][r] = route_d
        dp = [[math.inf] * (truck_count + 1) for _ in range(m + 1)]
        choice = [[-1] * (truck_count + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0
        for i in range(1, m + 1):
            for t in range(1, min(i, truck_count) + 1):
                best_val = math.inf
                best_j = -1
                for j in range(t - 1, i):
                    if dp[j][t-1] < math.inf:
                        cand = max(dp[j][t-1], seg_dist[j][i])
                        if cand < best_val or (cand == best_val and j < best_j):
                            best_val = cand
                            best_j = j
                dp[i][t] = best_val
                choice[i][t] = best_j
        routes = []
        i = m
        t = truck_count
        while t > 0:
            j = choice[i][t]
            seg = tour[j:i]
            routes.append([0] + seg + [0])
            i = j
            t -= 1
        routes.reverse()
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # Initialize random number generator for reproducibility
    rng = random.Random(42)

    # Generate nearest-neighbor tour
    visited = [False] * n
    visited[0] = True
    current = 0
    tour_nn = []
    for _ in range(m):
        best = -1
        best_dist = math.inf
        for v in range(1, n):
            if not visited[v]:
                d = distance_matrix[current, v]
                if d < best_dist or (d == best_dist and v < best):
                    best_dist = d
                    best = v
        tour_nn.append(best)
        visited[best] = True
        current = best

    # Generate initial population
    pop_size = 50
    population = [tour_nn]
    while len(population) < pop_size:
        perm = list(range(1, n))
        rng.shuffle(perm)
        population.append(perm)

    # Evaluate initial fitness
    fitness = []
    best_routes = None
    best_max = math.inf
    for perm in population:
        routes = tour_to_routes(perm)
        max_d = max(route_dist(r) for r in routes)
        fitness.append(max_d)
        if max_d < best_max:
            best_max = max_d
            best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)

    # GA parameters
    max_generations = 100
    crossover_rate = 0.8
    mutation_rate = 0.1
    elite_size = 2

    def crossover(p1, p2):
        size = len(p1)
        a = rng.randint(0, size - 2)
        b = rng.randint(a + 1, size - 1)
        child1 = [None] * size
        child1[a:b] = p1[a:b]
        child2 = [None] * size
        child2[a:b] = p2[a:b]
        # Fill remaining positions
        idx1 = 0
        idx2 = 0
        for i in range(size):
            if child1[i] is None:
                while p2[idx1] in child1:
                    idx1 += 1
                child1[i] = p2[idx1]
                idx1 += 1
            if child2[i] is None:
                while p1[idx2] in child2:
                    idx2 += 1
                child2[i] = p1[idx2]
                idx2 += 1
        return child1, child2

    for gen in range(max_generations):
        # Tournament selection (size 3, deterministic tie-break by index)
        new_pop = []
        # Elitism: keep best individuals
        sorted_indices = sorted(range(pop_size), key=lambda i: (fitness[i], i))
        for idx in sorted_indices[:elite_size]:
            new_pop.append(population[idx][:])
        # Fill rest
        while len(new_pop) < pop_size:
            # Select first parent
            cands = [rng.randint(0, pop_size - 1) for _ in range(3)]
            best1 = min(cands, key=lambda c: (fitness[c], c))
            p1 = population[best1]
            # Select second parent
            cands = [rng.randint(0, pop_size - 1) for _ in range(3)]
            best2 = min(cands, key=lambda c: (fitness[c], c))
            p2 = population[best2]
            # Crossover
            if rng.random() < crossover_rate:
                child1, _ = crossover(p1, p2)
                child = child1
            else:
                child = p1[:]
            # Mutation: swap two customers
            if rng.random() < mutation_rate:
                idx1 = rng.randint(0, m - 1)
                idx2 = rng.randint(0, m - 1)
                child[idx1], child[idx2] = child[idx2], child[idx1]
            new_pop.append(child)

        # Evaluate new population
        pop = population
        new_fitness = []
        for perm in new_pop:
            routes = tour_to_routes(perm)
            max_d = max(route_dist(r) for r in routes)
            new_fitness.append(max_d)
            if max_d < best_max:
                best_max = max_d
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
        population = new_pop
        fitness = new_fitness

    return best_routes