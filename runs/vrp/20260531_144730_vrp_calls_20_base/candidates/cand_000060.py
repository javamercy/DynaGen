import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    if truck_count <= 0:
        return []
    customers = list(range(1, n))
    N = n - 1

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def two_opt(route):
        improved = True
        best = route[:]
        best_dist = route_dist(best)
        while improved:
            improved = False
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < best_dist - 1e-12:
                        best = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
        return best, best_dist

    def dp_split(giant_tour):
        m = len(giant_tour)
        if m == 0:
            return [[0,0] for _ in range(truck_count)], 0.0
        K = min(truck_count, m)
        seg = [[0.0]*m for _ in range(m)]
        for i in range(m):
            d = distance_matrix[0, giant_tour[i]]
            seg[i][i] = d + distance_matrix[giant_tour[i], 0]
            for j in range(i+1, m):
                d += distance_matrix[giant_tour[j-1], giant_tour[j]]
                seg[i][j] = d + distance_matrix[giant_tour[j], 0]
        INF = 1e15
        dp = [[INF] * (m+1) for _ in range(K+1)]
        parent = [[-1] * (m+1) for _ in range(K+1)]
        dp[0][0] = 0.0
        for k in range(1, K+1):
            for i in range(k, m+1):
                for j in range(k-1, i):
                    cand = max(dp[k-1][j], seg[j][i-1])
                    if cand < dp[k][i]:
                        dp[k][i] = cand
                        parent[k][i] = j
        best_max = dp[K][m]
        routes = []
        k = K
        i = m
        while k > 0:
            j = parent[k][i]
            if j == -1:
                break
            segment = [0] + giant_tour[j:i] + [0]
            routes.append(segment)
            i = j
            k -= 1
        routes.reverse()
        while len(routes) < truck_count:
            routes.append([0,0])
        return routes, best_max

    def nearest_neighbor_tour():
        unvisited = set(customers)
        current = 0
        tour = []
        while unvisited:
            min_dist = min(distance_matrix[current, c] for c in unvisited)
            candidates = [c for c in unvisited if distance_matrix[current, c] == min_dist]
            next_node = random.choice(candidates)
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return tour

    pop_size = min(50, max(10, N // 2))
    population = []
    seed_tour = nearest_neighbor_tour()
    population.append(seed_tour)
    for _ in range(pop_size - 1):
        tour = customers[:]
        random.shuffle(tour)
        population.append(tour)

    fitness = []
    best_routes = None
    best_max = float('inf')
    for tour in population:
        routes, maxd = dp_split(tour)
        fitness.append(maxd)
        if maxd < best_max - 1e-12:
            best_max = maxd
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
            for idx in range(truck_count):
                route = best_routes[idx]
                if len(route) > 2:
                    new_route, _ = two_opt(route)
                    best_routes[idx] = new_route
            new_max = max(route_dist(r) for r in best_routes)
            if new_max < best_max - 1e-12:
                best_max = new_max
                report_best_vrp(best_routes)

    generations = 100
    mutation_rate = 0.1
    tourn_size = 2

    for gen in range(generations):
        new_pop = []
        while len(new_pop) < pop_size:
            idx1 = random.sample(range(pop_size), tourn_size)
            idx2 = random.sample(range(pop_size), tourn_size)
            parent1 = min(idx1, key=lambda i: fitness[i])
            parent2 = min(idx2, key=lambda i: fitness[i])
            p1 = population[parent1]
            p2 = population[parent2]
            size = len(p1)
            start = random.randint(0, size-1)
            end = random.randint(start+1, size)
            child = [None]*size
            child[start:end] = p1[start:end]
            remaining = [c for c in p2 if c not in child[start:end]]
            idx = 0
            for i in range(size):
                if child[i] is None:
                    child[i] = remaining[idx]
                    idx += 1
            if random.random() < mutation_rate:
                i, j = random.sample(range(size), 2)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        new_fitness = []
        for tour in new_pop:
            routes, maxd = dp_split(tour)
            new_routes = []
            for route in routes:
                if len(route) > 2:
                    new_route, _ = two_opt(route)
                    new_routes.append(new_route)
                else:
                    new_routes.append(route[:])
            new_max = max(route_dist(r) for r in new_routes)
            new_fitness.append(new_max)
            if new_max < best_max - 1e-12:
                best_max = new_max
                best_routes = [r[:] for r in new_routes]
                report_best_vrp(best_routes)
        best_idx = min(range(pop_size), key=lambda i: fitness[i])
        if fitness[best_idx] < min(new_fitness):
            worst_idx = max(range(pop_size), key=lambda i: new_fitness[i])
            new_pop[worst_idx] = population[best_idx][:]
            new_fitness[worst_idx] = fitness[best_idx]
        population = new_pop
        fitness = new_fitness
    return best_routes