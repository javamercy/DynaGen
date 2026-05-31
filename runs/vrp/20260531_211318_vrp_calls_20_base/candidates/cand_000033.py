import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix
    num_customers = n - 1
    customers = list(range(1, n))

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    # Split procedure for a permutation of customers (minimizes max route distance)
    def split(perm):
        m = len(perm)
        if m == 0:
            return [[0, 0] for _ in range(truck_count)], 0.0
        # Precompute segment distances
        seg = [[0.0]*m for _ in range(m)]
        for i in range(m):
            for j in range(i, m):
                if i == j:
                    seg[i][j] = dist[0][perm[i]] + dist[perm[i]][0]
                else:
                    seg[i][j] = seg[i][j-1] - dist[perm[j-1]][0] + dist[perm[j-1]][perm[j]] + dist[perm[j]][0]
        # DP: dp[k][i] = min max distance for first i+1 customers with k routes
        INF = float('inf')
        dp = [[INF]*m for _ in range(truck_count+1)]
        prev = [[-1]*m for _ in range(truck_count+1)]
        for k in range(1, truck_count+1):
            for i in range(m):
                best = INF
                best_j = -1
                # try starting from j-1 as previous end
                for j in range(-1, i):  # j = -1 means no previous customers
                    if j == -1:
                        prev_max = 0.0
                    else:
                        prev_max = dp[k-1][j]
                    if prev_max >= INF:
                        continue
                    seg_val = seg[j+1][i]  # segment from j+1 to i
                    candidate = prev_max if prev_max > seg_val else seg_val
                    if candidate < best:
                        best = candidate
                        best_j = j
                dp[k][i] = best
                prev[k][i] = best_j
        # Find best k (up to truck_count)
        best_max = INF
        best_k = -1
        for k in range(1, truck_count+1):
            if dp[k][m-1] < best_max:
                best_max = dp[k][m-1]
                best_k = k
        if best_k == -1:
            # fallback: assign all to one route
            route = [0] + perm + [0]
            return [route] + [[0,0] for _ in range(truck_count-1)], route_distance(route)
        # Reconstruct routes
        routes = []
        i = m-1
        k = best_k
        while k > 0:
            j = prev[k][i]
            if j == -1:
                seg_start = 0
            else:
                seg_start = j+1
            seg_cust = perm[seg_start:i+1]
            route = [0] + seg_cust + [0]
            routes.insert(0, route)
            i = j
            k -= 1
        # Add empty routes
        while len(routes) < truck_count:
            routes.append([0,0])
        return routes, best_max

    # Initial population: random permutations via random keys
    pop_size = 20
    population = []
    for _ in range(pop_size):
        keys = [random.random() for _ in range(num_customers)]
        perm = [c for _,c in sorted(zip(keys, customers))]
        routes, max_val = split(perm)
        total = sum(route_distance(r) for r in routes)
        population.append((keys, routes, max_val, total))

    # Best solution tracking
    best_routes = min(population, key=lambda x: (x[2], x[3]))[1]
    best_max = max(route_distance(r) for r in best_routes)
    best_total = sum(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    # GA parameters
    generations = 30
    tourn_size = 2
    crossover_rate = 0.8
    mutation_rate = 1.0 / num_customers if num_customers > 0 else 0.0

    for gen in range(generations):
        new_population = []
        # Elitism: keep best 2
        population.sort(key=lambda x: (x[2], x[3]))
        new_population.extend(population[:2])

        while len(new_population) < pop_size:
            # Tournament selection
            parents = []
            for _ in range(2):
                tourn = random.sample(population, tourn_size)
                parent = min(tourn, key=lambda x: (x[2], x[3]))
                parents.append(parent)
            parent1_keys, parent1_routes, _, _ = parents[0]
            parent2_keys, parent2_routes, _, _ = parents[1]

            # Crossover: uniform
            if random.random() < crossover_rate:
                child_keys = []
                for i in range(num_customers):
                    if random.random() < 0.5:
                        child_keys.append(parent1_keys[i])
                    else:
                        child_keys.append(parent2_keys[i])
            else:
                child_keys = parent1_keys[:]

            # Mutation: Gaussian perturbation
            for i in range(num_customers):
                if random.random() < mutation_rate:
                    child_keys[i] += random.gauss(0, 0.1)
                    if child_keys[i] < 0:
                        child_keys[i] = 0.0
                    elif child_keys[i] > 1:
                        child_keys[i] = 1.0

            # Decode
            perm = [c for _,c in sorted(zip(child_keys, customers))]
            routes, max_val = split(perm)
            total = sum(route_distance(r) for r in routes)

            # Local search: 2-opt on each route
            improved = True
            ls_iter = 0
            while improved and ls_iter < num_customers:
                improved = False
                for t in range(truck_count):
                    route = routes[t]
                    if len(route) <= 3:
                        continue
                    best_improv = 0.0
                    best_ij = None
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            if j == i+1:
                                continue
                            # reverse segment i..j
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_dist = 0.0
                            for k in range(len(new_route)-1):
                                new_dist += dist[new_route[k], new_route[k+1]]
                            old_dist = 0.0
                            for k in range(len(route)-1):
                                old_dist += dist[route[k], route[k+1]]
                            delta = old_dist - new_dist
                            if delta > best_improv + 1e-9:
                                best_improv = delta
                                best_ij = (i, j, new_route)
                    if best_ij is not None:
                        i, j, new_route = best_ij
                        routes[t] = new_route
                        improved = True
                ls_iter += 1
            # Re-evaluate after local search
            max_val = max(route_distance(r) for r in routes)
            total = sum(route_distance(r) for r in routes)

            new_population.append((child_keys, routes, max_val, total))

        population = new_population

        # Update best
        for _, routes, max_val, total in population:
            if max_val < best_max - 1e-9 or (abs(max_val - best_max) < 1e-9 and total < best_total):
                best_max = max_val
                best_total = total
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)

    return best_routes