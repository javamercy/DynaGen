import random
import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    if truck_count >= n:
        routes = []
        for i in range(truck_count):
            if i < len(customers):
                routes.append([0, customers[i], 0])
            else:
                routes.append([0, 0])
        return routes

    def route_cost(route):
        if len(route) <= 2:
            return 0.0
        cost = 0.0
        for i in range(len(route) - 1):
            cost += distance_matrix[route[i], route[i+1]]
        return cost

    # DP function to split a permutation into exactly truck_count routes minimizing max route cost
    def split_perm(perm):
        m = len(perm)
        if m == 0:
            return [[0, 0] for _ in range(truck_count)]
        # precompute segment costs: seg_cost[i][j] = cost of route covering perm[i:j] (i inclusive, j exclusive)
        seg_cost = [[0.0] * (m + 1) for _ in range(m)]
        for i in range(m):
            cost = 0.0
            for j in range(i + 1, m + 1):
                if j == i + 1:
                    # single customer: distance depot->cust + cust->depot
                    cost = distance_matrix[depot, perm[i]] + distance_matrix[perm[i], depot]
                else:
                    # extend from j-1 to j-2? Actually for interval [i, j-1], we need cost from depot to first, then between, then last to depot.
                    # Recompute properly each time to avoid errors: simpler to compute fresh each i, j
                    pass
        # Actually compute seg_cost correctly:
        # seg_cost[i][j] = cost of route [0] + perm[i:j] + [0] (including depot at both ends)
        for i in range(m):
            for j in range(i + 1, m + 1):
                cost = distance_matrix[depot, perm[i]]
                for k in range(i, j - 1):
                    cost += distance_matrix[perm[k], perm[k+1]]
                cost += distance_matrix[perm[j-1], depot]
                seg_cost[i][j] = cost
        # DP: dp[k][i] = min max cost for first i customers with k routes, where i is count of customers in prefix
        INF = float('inf')
        dp = [[INF] * (m + 1) for _ in range(truck_count + 1)]
        dp[0][0] = 0.0
        # backtrack: prev[k][i] = (k-1, j) such that dp[k][i] = max(dp[k-1][j], seg_cost[j][i])
        prev = [[None] * (m + 1) for _ in range(truck_count + 1)]
        for k in range(1, truck_count + 1):
            for i in range(m + 1):
                best = INF
                best_j = -1
                for j in range(i + 1):  # j is number of customers before the new route
                    if dp[k-1][j] < INF:
                        if i == j:
                            cur = dp[k-1][j]  # empty route
                        else:
                            cur = max(dp[k-1][j], seg_cost[j][i-1])  # careful: seg_cost indices
                        # Actually seg_cost[j][i] expects interval [j, i) where j and i are indices in perm
                        # j is count of customers before new route, so the new route covers perm[j:i]
                        if i > j:
                            cur = max(dp[k-1][j], seg_cost[j][i-1])
                        else:
                            cur = dp[k-1][j]
                        if cur < best:
                            best = cur
                            best_j = j
                dp[k][i] = best
                prev[k][i] = (k-1, best_j)
        # reconstruction
        k = truck_count
        i = m
        routes_list = []
        while k > 0:
            _, j = prev[k][i]
            # route covers customers from j to i-1 (if j < i)
            route = [depot]
            if j < i:
                for idx in range(j, i):
                    route.append(perm[idx])
            route.append(depot)
            routes_list.append(route)
            k -= 1
            i = j
        routes_list.reverse()
        # fill remaining routes if truck_count > m
        while len(routes_list) < truck_count:
            routes_list.append([depot, depot])
        # return routes and the max cost
        max_cost = max(route_cost(r) for r in routes_list)
        return routes_list, max_cost

    # Genetic Algorithm parameters
    pop_size = max(50, n * 2)
    generations = max(100, n * 2)
    crossover_rate = 0.8
    mutation_rate = 0.2
    tournament_size = 3
    elitism_count = 2

    # Generate initial population: random permutations
    population = []
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes, cost = split_perm(perm)
        population.append((perm, routes, cost))
    # Sort by cost ascending
    population.sort(key=lambda x: x[2])
    best_perm, best_routes, best_cost = population[0]
    report_best_vrp(best_routes)

    # GA loop
    for gen in range(generations):
        new_population = []
        # Elitism
        for i in range(elitism_count):
            new_population.append(population[i])
        while len(new_population) < pop_size:
            # Tournament selection
            candidates = random.sample(population, tournament_size)
            candidates.sort(key=lambda x: x[2])
            parent1 = candidates[0][0][:]
            # second parent: also tournament, avoid same?
            candidates2 = random.sample(population, tournament_size)
            candidates2.sort(key=lambda x: x[2])
            parent2 = candidates2[0][0][:]
            # Crossover: Order crossover (OX)
            if random.random() < crossover_rate:
                # Choose two crossover points
                pt1 = random.randint(0, n-2)
                pt2 = random.randint(pt1+1, n-1)
                child = [None] * (n-1)
                # Copy segment from parent1
                child[pt1:pt2] = parent1[pt1:pt2]
                # Fill remaining from parent2 in order
                idx = pt2
                for c in parent2:
                    if c not in child:
                        if idx >= n-1:
                            idx = 0
                        child[idx] = c
                        idx += 1
                child_perm = child
            else:
                child_perm = parent1
            # Mutation
            if random.random() < mutation_rate:
                if random.random() < 0.5:
                    # Swap two random positions
                    i1, i2 = random.sample(range(len(child_perm)), 2)
                    child_perm[i1], child_perm[i2] = child_perm[i2], child_perm[i1]
                else:
                    # Invert a subsequence
                    i1 = random.randint(0, len(child_perm)-2)
                    i2 = random.randint(i1+2, len(child_perm))
                    child_perm[i1:i2] = reversed(child_perm[i1:i2])
            # Evaluate
            child_routes, child_cost = split_perm(child_perm)
            new_population.append((child_perm, child_routes, child_cost))
        # Sort new population
        new_population.sort(key=lambda x: x[2])
        population = new_population
        # Update best
        if population[0][2] < best_cost:
            best_perm, best_routes, best_cost = population[0]
            report_best_vrp(best_routes)
    # Return best routes
    return best_routes