import numpy as np
import random
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    n_cust = n - 1
    if truck_count <= 0:
        return []

    def compute_route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d

    # Greedy construction (from parent) to get initial routes
    def greedy_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        unassigned.sort(key=lambda c: -distance_matrix[0][c])
        route_dists = [0.0 for _ in routes]
        for cust in unassigned:
            best_new_max = float('inf')
            best_route_idx = -1
            best_pos = -1
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    succ = route[pos]
                    increase = distance_matrix[prev][cust] + distance_matrix[cust][succ] - distance_matrix[prev][succ]
                    new_route_dist = route_dists[r_idx] + increase
                    new_max = new_route_dist
                    for other_idx, d in enumerate(route_dists):
                        if other_idx != r_idx and d > new_max:
                            new_max = d
                    if new_max < best_new_max or (new_max == best_new_max and r_idx < best_route_idx):
                        best_new_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
            route = routes[best_route_idx]
            route.insert(best_pos, cust)
            route_dists[best_route_idx] = compute_route_dist(route)
        return routes, route_dists

    # Split a permutation into exactly k routes minimizing max distance
    def split_permutation(perm, k):
        n_c = len(perm)
        if k > n_c:
            # Not enough customers; split into n_c routes and pad with empty
            routes, max_val = split_permutation(perm, n_c)
            while len(routes) < k:
                routes.append([0, 0])
            return routes, max_val
        # Precompute prefix sum of distances along permutation
        pref = [0.0] * (n_c)  # pref[i] = sum_{t=0}^{i-1} dist(perm[t], perm[t+1]) for i=0..n_c-1, pref[0]=0
        for i in range(1, n_c):
            pref[i] = pref[i-1] + distance_matrix[perm[i-1]][perm[i]]
        # Helper to get segment distance from s to e (exclusive)
        def seg_dist(s, e):
            if s == e:
                return 0.0
            first = distance_matrix[0][perm[s]]
            last = distance_matrix[perm[e-1]][0]
            middle = pref[e-1] - pref[s]
            return first + middle + last
        INF = 1e100
        dp = [[INF]*(k+1) for _ in range(n_c+1)]
        choice = [[-1]*(k+1) for _ in range(n_c+1)]
        dp[0][0] = 0.0
        for i in range(1, n_c+1):
            max_j = min(i, k)
            for j in range(1, max_j+1):
                best = INF
                best_h = -1
                for h in range(i):
                    if dp[h][j-1] >= INF/2:
                        continue
                    seg = seg_dist(h, i)
                    new_max = max(dp[h][j-1], seg)
                    if new_max < best - 1e-12:
                        best = new_max
                        best_h = h
                dp[i][j] = best
                choice[i][j] = best_h
        # Reconstruct
        routes = []
        i = n_c
        j = k
        segments = []
        while j > 0:
            h = choice[i][j]
            segments.append((h, i))
            i = h
            j -= 1
        segments.reverse()
        for s, e in segments:
            route = [0] + perm[s:e] + [0]
            routes.append(route)
        return routes, dp[n_c][k]

    # Genetic Algorithm
    pop_size = 10
    max_gen = 100
    elite_size = 1
    mutation_prob = 0.2
    crossover_prob = 0.8
    # Initial population
    population = []
    # Greedy individual
    greedy_routes, _ = greedy_construction()
    # Convert to permutation
    greedy_perm = []
    for route in greedy_routes:
        for cust in route[1:-1]:
            greedy_perm.append(cust)
    population.append(greedy_perm)
    # Random individuals
    for _ in range(pop_size - 1):
        perm = list(range(1, n))
        random.shuffle(perm)
        population.append(perm)

    # Evaluate individuals
    def evaluate(perm):
        k = min(truck_count, len(perm))
        routes, max_val = split_permutation(perm, k)
        # If k < truck_count, pad with empty routes
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes, max_val

    # Find best initial
    best_routes = None
    best_max = float('inf')
    for perm in population:
        routes, max_val = evaluate(perm)
        if max_val < best_max - 1e-12:
            best_max = max_val
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    # Order crossover (OX1)
    def order_crossover(p1, p2):
        size = len(p1)
        a = random.randint(0, size-1)
        b = random.randint(0, size-1)
        if a > b:
            a, b = b, a
        child = [None]*size
        # Copy segment from parent1
        child[a:b+1] = p1[a:b+1]
        # Fill remaining from parent2 in order
        pos = (b+1) % size
        for i in range(size):
            idx = (b+1+i) % size
            if p2[idx] not in child:
                child[pos] = p2[idx]
                pos = (pos+1) % size
        return child

    # Mutation: swap two elements
    def swap_mutation(perm, prob):
        if random.random() < prob:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    # Inversion mutation
    def inversion_mutation(perm, prob):
        if random.random() < prob:
            i, j = random.sample(range(len(perm)), 2)
            if i > j:
                i, j = j, i
            perm[i:j+1] = reversed(perm[i:j+1])
        return perm

    # Tournament selection
    def tournament(pop, fits, size=2):
        idx = random.sample(range(len(pop)), size)
        best_idx = idx[0]
        for i in idx[1:]:
            if fits[i] < fits[best_idx]:
                best_idx = i
        return pop[best_idx]

    # Generations
    no_improve = 0
    for gen in range(max_gen):
        # Evaluate fitness (max distance) for all
        fits = []
        for perm in population:
            _, max_val = evaluate(perm)
            fits.append(max_val)
        # New population
        new_pop = []
        # Elitism
        sorted_idx = sorted(range(len(population)), key=lambda i: fits[i])
        for i in range(elite_size):
            new_pop.append(population[sorted_idx[i]][:])
        while len(new_pop) < pop_size:
            p1 = tournament(population, fits)
            p2 = tournament(population, fits)
            if random.random() < crossover_prob:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            child = swap_mutation(child, 0.2)
            child = inversion_mutation(child, 0.1)
            new_pop.append(child)
        population = new_pop
        # Evaluate and update best
        improved = False
        for perm in population:
            routes, max_val = evaluate(perm)
            if max_val < best_max - 1e-12:
                best_max = max_val
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
                improved = True
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 30:
                break

    # Return best found
    return best_routes