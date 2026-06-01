import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    # Helper functions
    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))

    def compute_max(routes):
        return max(route_dist(r) for r in routes)

    # DP split: given a permutation of customers (list excl depot), split into truck_count routes minimizing max distance
    def split_tour(tour):
        m = len(tour)
        if m == 0:
            return [[0, 0] for _ in range(truck_count)]
        # segment distances
        seg_dist = [[0] * (m + 1) for _ in range(m)]
        for l in range(m):
            acc = distance_matrix[0][tour[l]]
            for r in range(l + 1, m + 1):
                if r > l + 1:
                    acc += distance_matrix[tour[r - 2]][tour[r - 1]]
                if r == l + 1:
                    seg_dist[l][r] = distance_matrix[0][tour[l]] + distance_matrix[tour[l]][0]
                else:
                    seg_dist[l][r] = acc + distance_matrix[tour[r - 1]][0]
        # DP
        dp = [[math.inf] * (truck_count + 1) for _ in range(m + 1)]
        choice = [[-1] * (truck_count + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(1, m + 1):
            for t in range(1, min(i, truck_count) + 1):
                best_val = math.inf
                best_j = -1
                for j in range(t - 1, i):
                    if dp[j][t - 1] < math.inf:
                        cand = max(dp[j][t - 1], seg_dist[j][i])
                        if cand < best_val or (cand == best_val and j < best_j):
                            best_val = cand
                            best_j = j
                dp[i][t] = best_val
                choice[i][t] = best_j
        # reconstruct
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

    # Generate initial TSP tour (nearest neighbor)
    def nearest_neighbor_tour():
        tour = []
        visited = [False] * n
        visited[0] = True
        current = 0
        for _ in range(m):
            best = -1
            best_dist = math.inf
            for v in range(1, n):
                if not visited[v]:
                    d = distance_matrix[current][v]
                    if d < best_dist or (d == best_dist and v < best):
                        best_dist = d
                        best = v
            tour.append(best)
            visited[best] = True
            current = best
        return tour

    # TSP+DP solution
    tsp_tour = nearest_neighbor_tour()
    tsp_routes = split_tour(tsp_tour)
    best_routes = [list(r) for r in tsp_routes]
    best_max = compute_max(best_routes)
    report_best_vrp(best_routes)

    # GA parameters
    pop_size = min(20, n)
    generations = min(30, n)
    crossover_rate = 0.8
    mutation_rate = 0.1
    tournament_size = 3

    # Helper to encode a solution into a permutation: concatenate customers in order of routes (excluding depots)
    def encode(routes):
        perm = []
        for r in routes:
            for x in r:
                if x != 0:
                    perm.append(x)
        return perm

    # Initial population: include TSP+DP solution and random permutations
    population = []
    # add TSP+DP solution as first individual
    pop_perm = encode(tsp_routes)
    pop_routes = tsp_routes
    population.append((pop_perm, pop_routes, compute_max(pop_routes)))
    # generate random permutations for rest
    for _ in range(pop_size - 1):
        perm = customers[:]
        random.shuffle(perm)
        routes = split_tour(perm)
        cost = compute_max(routes)
        population.append((perm, routes, cost))
        if cost < best_max:
            best_max = cost
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    # Order crossover
    def crossover_ox(p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        ptr = 0
        for gene in p2:
            if gene not in child:
                while ptr < a or ptr > b:
                    ptr += 1
                child[ptr] = gene
                ptr += 1
        return child

    # Swap mutation
    def mutate(perm):
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
        return perm

    # Tournament selection
    def tournament(pop, k):
        selected = random.sample(pop, k)
        selected.sort(key=lambda x: x[2])  # cost
        return selected[0][0]  # return permutation

    # Main GA loop
    for gen in range(generations):
        new_pop = []
        # Elitism: keep best individual
        best_ind = min(population, key=lambda x: x[2])
        new_pop.append(best_ind)
        while len(new_pop) < pop_size:
            # selection
            p1 = tournament(population, tournament_size)
            p2 = tournament(population, tournament_size)
            # crossover
            if random.random() < crossover_rate:
                child_perm = crossover_ox(p1, p2)
            else:
                child_perm = p1[:]
            # mutation
            if random.random() < mutation_rate:
                child_perm = mutate(child_perm)
            # decode
            child_routes = split_tour(child_perm)
            child_cost = compute_max(child_routes)
            new_pop.append((child_perm, child_routes, child_cost))
            if child_cost < best_max:
                best_max = child_cost
                best_routes = [list(r) for r in child_routes]
                report_best_vrp(best_routes)
        population = new_pop

    return best_routes