import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def two_opt(route, max_iter=10):
        route = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
        return route

    def segment_cost(start_idx, end_idx, perm):
        # cost of route: depot -> perm[start_idx] -> ... -> perm[end_idx] -> depot
        if start_idx > end_idx:
            return 0.0
        cost = distance_matrix[0, perm[start_idx]]
        for i in range(start_idx, end_idx):
            cost += distance_matrix[perm[i], perm[i+1]]
        cost += distance_matrix[perm[end_idx], 0]
        return cost

    def split_permutation(perm):
        m = len(perm)
        INF = 1e100
        # dp[i][k] = min max distance for first i customers (i from 0 to m) using k routes
        dp = [[INF] * (truck_count + 1) for _ in range(m + 1)]
        prev = [[-1] * (truck_count + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0
        for i in range(1, m + 1):
            for k in range(1, min(i, truck_count) + 1):
                best_val = INF
                best_j = -1
                # j is the index where the last route starts (1-indexed customer positions)
                for j in range(k - 1, i):
                    seg = segment_cost(j, i - 1, perm)  # customers from j to i-1
                    if dp[j][k-1] != INF:
                        candidate = max(dp[j][k-1], seg)
                        if candidate < best_val:
                            best_val = candidate
                            best_j = j
                dp[i][k] = best_val
                prev[i][k] = best_j
        if dp[m][truck_count] == INF:
            # fallback: use greedy last routes
            return None, None
        # reconstruct routes
        routes = []
        i = m
        k = truck_count
        while k > 0:
            j = prev[i][k]
            route = [0] + [perm[idx] for idx in range(j, i)] + [0]
            routes.append(route)
            i = j
            k -= 1
        routes.reverse()
        # Fill missing trucks if any (should not happen)
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes, dp[m][truck_count]

    def evaluate(perm):
        routes, max_dist = split_permutation(perm)
        if routes is None:
            return None, None, None
        total_dist = sum(route_distance(r) for r in routes)
        return routes, max_dist, total_dist

    def generate_random_perm():
        perm = customers[:]
        random.shuffle(perm)
        return perm

    def ordered_crossover(p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[a:b+1] = p1[a:b+1]
        pos = (b + 1) % size
        for city in p2:
            if city not in child:
                child[pos] = city
                pos = (pos + 1) % size
        return child

    def swap_mutation(perm, prob=0.1):
        if random.random() < prob:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    # Genetic Algorithm parameters
    pop_size = 30
    generations = min(500, n * truck_count * 2)
    tournament_size = 2
    elitism = 1

    # Initialize population
    population = [generate_random_perm() for _ in range(pop_size)]
    best_routes = None
    best_max = float('inf')
    best_total = float('inf')

    for perm in population:
        routes, max_dist, total_dist = evaluate(perm)
        if routes is not None and (max_dist < best_max or (max_dist == best_max and total_dist < best_total)):
            best_max = max_dist
            best_total = total_dist
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    for gen in range(generations):
        # Evaluate fitness (minimize max distance, then total)
        fitness = []
        for perm in population:
            _, max_dist, total_dist = evaluate(perm)
            if max_dist is None:
                fitness.append((float('inf'), float('inf')))
            else:
                fitness.append((max_dist, total_dist))
        # Elitism: keep best solution
        sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])
        new_population = [population[i][:] for i in sorted_indices[:elitism]]
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            idx1 = random.sample(range(pop_size), tournament_size)
            idx2 = random.sample(range(pop_size), tournament_size)
            parent1_idx = min(idx1, key=lambda i: fitness[i])
            parent2_idx = min(idx2, key=lambda i: fitness[i])
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            child = ordered_crossover(parent1, parent2)
            child = swap_mutation(child, prob=0.1)
            new_population.append(child)
        population = new_population
        # Update best
        for perm in population:
            routes, max_dist, total_dist = evaluate(perm)
            if routes is not None and (max_dist < best_max or (max_dist == best_max and total_dist < best_total)):
                best_max = max_dist
                best_total = total_dist
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)

    # Final improvement: apply 2-opt to each route of best solution
    if best_routes is not None:
        for idx in range(len(best_routes)):
            if len(best_routes[idx]) > 2:
                best_routes[idx] = two_opt(best_routes[idx], max_iter=20)
        best_max = max(route_distance(r) for r in best_routes)
        # Recalculate best_max after 2-opt (optional report)
        if best_max < best_max:  # always false, but keep structure
            pass  # already best
    return best_routes