import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    
    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    # Nearest-neighbor construction for initial giant tour
    def nearest_neighbor_tour():
        unvisited = set(customers)
        current = 0
        tour = []
        while unvisited:
            next_cust = min(unvisited, key=lambda c: distance_matrix[current, c])
            tour.append(next_cust)
            unvisited.remove(next_cust)
            current = next_cust
        return tour
    
    # DP splitting: given a permutation of customers, compute optimal min-max split
    def split(perm):
        m = len(perm)
        if m == 0:
            return [[0, 0] for _ in range(truck_count)], 0.0
        # Precompute segment distances: seg[i][j] = route distance for segment i..j inclusive
        seg = [[0.0] * m for _ in range(m)]
        for i in range(m):
            total = distance_matrix[0, perm[i]]
            for j in range(i, m):
                if j > i:
                    total += distance_matrix[perm[j-1], perm[j]]
                seg[i][j] = total + distance_matrix[perm[j], 0]
        max_routes = min(truck_count, m)
        INF = 1e100
        dp = [[INF] * (max_routes + 1) for _ in range(m + 1)]
        split_point = [[-1] * (max_routes + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0
        for i in range(1, m + 1):
            for k in range(1, max_routes + 1):
                for j in range(i):
                    seg_len = seg[j][i-1]
                    max_val = max(dp[j][k-1], seg_len)
                    if dp[i][k] > max_val:
                        dp[i][k] = max_val
                        split_point[i][k] = j
        best_max = dp[m][max_routes]
        # Reconstruct routes
        routes = []
        pos = m
        for k in range(max_routes, 0, -1):
            start = split_point[pos][k]
            segment = perm[start:pos]
            route = [0] + segment + [0]
            routes.append(route)
            pos = start
        routes.reverse()
        # Add empty routes if needed
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes, best_max
    
    # Genetic algorithm operators
    def order_crossover(p1, p2):
        size = len(p1)
        a = random.randint(0, size - 1)
        b = random.randint(0, size - 1)
        if a > b:
            a, b = b, a
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        pos = (b + 1) % size
        for c in p2:
            if c not in child:
                child[pos] = c
                pos = (pos + 1) % size
        return child
    
    def swap_mutation(perm):
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
    
    # Evaluate fitness: returns (max_dist, routes)
    def evaluate(perm):
        routes, max_dist = split(perm)
        return max_dist, routes
    
    # Population initialization
    pop_size = 20
    generations = 100
    mutation_rate = 0.1
    crossover_rate = 0.8
    elitism = 1
    
    pop = []
    nn_tour = nearest_neighbor_tour()
    pop.append(nn_tour)
    for _ in range(pop_size - 1):
        perm = random.sample(customers, len(customers))
        pop.append(perm)
    
    # Evaluate initial population
    fitness = [evaluate(p) for p in pop]
    best_idx = min(range(pop_size), key=lambda i: (fitness[i][0], i))
    best_fitness, best_routes = fitness[best_idx]
    report_best_vrp(best_routes)
    
    for gen in range(generations):
        new_pop = []
        # Elitism
        sorted_indices = sorted(range(pop_size), key=lambda i: (fitness[i][0], i))
        for i in sorted_indices[:elitism]:
            new_pop.append(pop[i][:])  # copy
        while len(new_pop) < pop_size:
            # Binary tournament
            i1, i2 = random.sample(range(pop_size), 2)
            if fitness[i1][0] < fitness[i2][0]:
                p1 = pop[i1]
            elif fitness[i2][0] < fitness[i1][0]:
                p1 = pop[i2]
            else:
                p1 = pop[i1] if i1 < i2 else pop[i2]
            i1, i2 = random.sample(range(pop_size), 2)
            if fitness[i1][0] < fitness[i2][0]:
                p2 = pop[i1]
            elif fitness[i2][0] < fitness[i1][0]:
                p2 = pop[i2]
            else:
                p2 = pop[i1] if i1 < i2 else pop[i2]
            # Crossover
            if random.random() < crossover_rate:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            # Mutation
            if random.random() < mutation_rate:
                swap_mutation(child)
            new_pop.append(child)
        pop = new_pop
        # Evaluate new population
        fitness = [evaluate(p) for p in pop]
        # Update best
        for i, (f, r) in enumerate(fitness):
            if f < best_fitness or (f == best_fitness and i < best_idx):
                best_fitness = f
                best_routes = r
                best_idx = i
                report_best_vrp(best_routes)
    
    # Post-processing: 2-opt improvement on each route of best solution
    def two_opt(route, max_iter=10):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
        return route
    
    for idx in range(truck_count):
        if len(best_routes[idx]) > 2:
            best_routes[idx] = two_opt(best_routes[idx])
    new_max = max(route_distance(r) for r in best_routes)
    if new_max < best_fitness:
        report_best_vrp(best_routes)
    
    return best_routes