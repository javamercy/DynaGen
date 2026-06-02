import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0,0] for _ in range(truck_count)]
    customers = list(range(1, n))
    random.seed(0)

    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def evaluate(perm):
        m = len(perm)
        if m == 0:
            return 0.0, [[0,0] for _ in range(truck_count)]
        # precompute for O(1) segment distance
        first_depot = np.array([distance_matrix[0, c] for c in perm])
        last_depot = np.array([distance_matrix[c, 0] for c in perm])
        int_prefix = np.zeros(m, dtype=float)
        for i in range(1, m):
            int_prefix[i] = int_prefix[i-1] + distance_matrix[perm[i-1], perm[i]]
        INF = float('inf')
        dp = [[INF]*(truck_count+1) for _ in range(m+1)]
        pred = [[None]*(truck_count+1) for _ in range(m+1)]
        for t in range(truck_count+1):
            dp[0][t] = 0.0
        for i in range(1, m+1):
            for t in range(1, truck_count+1):
                best = dp[i][t-1]
                best_j = None
                for j in range(i):
                    seg = first_depot[j] + (int_prefix[i-1] - int_prefix[j]) + last_depot[i-1]
                    cand = max(dp[j][t-1], seg)
                    if cand < best - 1e-12:
                        best = cand
                        best_j = j
                dp[i][t] = best
                pred[i][t] = best_j
        best_max = dp[m][truck_count]
        # reconstruct
        segments = []
        i = m
        t = truck_count
        while i > 0:
            j = pred[i][t]
            if j is None:
                segments.append(None)
                t -= 1
            else:
                segments.append((j, i-1))
                i = j
                t -= 1
        for _ in range(t):
            segments.append(None)
        segments.reverse()
        routes = []
        for seg in segments:
            if seg is None:
                routes.append([0,0])
            else:
                start, end = seg
                route = [0] + perm[start:end+1] + [0]
                routes.append(route)
        return best_max, routes

    pop_size = 20
    generations = 50
    mutation_rate = 0.1
    tournament_size = 3

    pop = [customers[:] for _ in range(pop_size)]
    for p in pop:
        random.shuffle(p)

    best_overall_routes = None
    best_overall_fitness = float('inf')

    fitnesses = []
    for perm in pop:
        fit, routes = evaluate(perm)
        fitnesses.append(fit)
        if fit < best_overall_fitness - 1e-12:
            best_overall_fitness = fit
            best_overall_routes = routes
            report_best_vrp(routes)

    for gen in range(generations):
        new_pop = []
        best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
        new_pop.append(pop[best_idx][:])
        while len(new_pop) < pop_size:
            idx1 = random.sample(range(pop_size), tournament_size)
            idx2 = random.sample(range(pop_size), tournament_size)
            winner1 = min(idx1, key=lambda i: fitnesses[i])
            winner2 = min(idx2, key=lambda i: fitnesses[i])
            parent1 = pop[winner1]
            parent2 = pop[winner2]
            size = len(parent1)
            a, b = sorted(random.sample(range(size), 2))
            child = [None]*size
            child[a:b+1] = parent1[a:b+1]
            remaining = [x for x in parent2 if x not in child[a:b+1]]
            idx = 0
            for i in range(size):
                if child[i] is None:
                    child[i] = remaining[idx]
                    idx += 1
            if random.random() < mutation_rate:
                i, j = random.sample(range(size), 2)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        pop = new_pop
        fitnesses = []
        for perm in pop:
            fit, routes = evaluate(perm)
            fitnesses.append(fit)
            if fit < best_overall_fitness - 1e-12:
                best_overall_fitness = fit
                best_overall_routes = routes
                report_best_vrp(routes)

    return best_overall_routes