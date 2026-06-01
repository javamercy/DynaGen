import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(42)  # deterministic random seed
    n = distance_matrix.shape[0]
    dist = distance_matrix
    customers = list(range(1, n))
    
    def route_dist(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d
    
    def max_dist(routes):
        return max(route_dist(r) for r in routes)
    
    def split_permutation(perm, K):
        M = len(perm)
        if M == 0:
            return [[0, 0] for _ in range(K)]
        # precompute cost[i][j] = distance of route covering perm[i..j]
        cost = [[0.0]*M for _ in range(M)]
        for i in range(M):
            s = dist[0][perm[i]]
            for j in range(i, M):
                if j > i:
                    s += dist[perm[j-1]][perm[j]]
                cost[i][j] = s + dist[perm[j]][0]
        INF = 1e100
        dp = [[INF]*(K+1) for _ in range(M)]
        split = [[-1]*(K+1) for _ in range(M)]
        for j in range(M):
            dp[j][1] = cost[0][j]
            split[j][1] = 0
        for t in range(2, K+1):
            for j in range(t-1, M):
                best = INF
                best_i = -1
                for i in range(t-1, j+1):
                    cand = max(dp[i-1][t-1], cost[i][j])
                    if cand < best:
                        best = cand
                        best_i = i
                dp[j][t] = best
                split[j][t] = best_i
        # reconstruct routes in order
        routes = []
        j = M-1
        for t in range(K, 0, -1):
            i = split[j][t]
            route_cust = perm[i:j+1]
            routes.append([0] + route_cust + [0])
            j = i-1
        routes.reverse()
        # fill empty routes if needed (should not happen)
        while len(routes) < K:
            routes.append([0,0])
        return routes
    
    def local_improve(routes, max_iter, stagnation_limit):
        best_routes = [list(r) for r in routes]
        best_max = max_dist(best_routes)
        report_best_vrp(best_routes)
        
        for iteration in range(max_iter):
            improved = False
            current_max = max_dist(routes)
            longest_indices = [i for i, r in enumerate(routes) if route_dist(r) == current_max]
            if not longest_indices:
                break
            r_idx = longest_indices[0]
            route = routes[r_idx]
            # Relocate from longest route
            for pos in range(1, len(route)-1):
                cust = route[pos]
                for other_idx, other_route in enumerate(routes):
                    if other_idx == r_idx:
                        continue
                    for other_pos in range(1, len(other_route)):
                        new_self = route[:pos] + route[pos+1:]
                        new_other = other_route[:other_pos] + [cust] + other_route[other_pos:]
                        new_routes = [list(r) for r in routes]
                        new_routes[r_idx] = new_self
                        new_routes[other_idx] = new_other
                        new_max = max_dist(new_routes)
                        if new_max < best_max * (1 - 1e-12):
                            best_max = new_max
                            best_routes = [list(r) for r in new_routes]
                            routes = new_routes
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-route swap
            for pos1 in range(1, len(routes[r_idx])-1):
                cust1 = routes[r_idx][pos1]
                for other_idx, other_route in enumerate(routes):
                    if other_idx == r_idx:
                        continue
                    for pos2 in range(1, len(other_route)-1):
                        cust2 = other_route[pos2]
                        new_route1 = routes[r_idx][:pos1] + [cust2] + routes[r_idx][pos1+1:]
                        new_route2 = other_route[:pos2] + [cust1] + other_route[pos2+1:]
                        new_routes = [list(r) for r in routes]
                        new_routes[r_idx] = new_route1
                        new_routes[other_idx] = new_route2
                        new_max = max_dist(new_routes)
                        if new_max < best_max * (1 - 1e-12):
                            best_max = new_max
                            best_routes = [list(r) for r in new_routes]
                            routes = new_routes
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt on each route
            for r_idx2, route2 in enumerate(routes):
                if len(route2) <= 3:
                    continue
                for i in range(1, len(route2)-2):
                    for j in range(i+1, len(route2)-1):
                        new_route = route2[:i] + route2[i:j+1][::-1] + route2[j+1:]
                        if route_dist(new_route) < route_dist(route2):
                            routes[r_idx2] = new_route
                            improved = True
                            current_max = max_dist(routes)
                            if current_max < best_max * (1 - 1e-12):
                                best_max = current_max
                                best_routes = [list(r) for r in routes]
                                report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return best_routes, best_max
    
    # Initialize population
    pop_size = 60
    generations = 40
    crossover_rate = 0.8
    mutation_rate = 0.1
    tournament_size = 3
    max_iter_local = n * truck_count
    stagnation_limit = max(10, n//10)
    
    population = []
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes = split_permutation(perm, truck_count)
        routes, fit = local_improve(routes, max_iter_local // 2, stagnation_limit)
        # reconstruct permutation from routes
        perm = []
        for r in routes:
            perm.extend(r[1:-1])
        population.append((fit, perm, routes))
    population.sort(key=lambda x: x[0])
    best_fitness = population[0][0]
    best_routes = population[0][2]
    
    # GA loop
    for gen in range(generations):
        new_pop = []
        for _ in range(pop_size // 2):
            # tournament selection
            candidates = random.sample(population, tournament_size)
            candidates.sort(key=lambda x: x[0])
            p1 = candidates[0]
            candidates = random.sample(population, tournament_size)
            candidates.sort(key=lambda x: x[0])
            p2 = candidates[0]
            # order crossover
            perm1, perm2 = p1[1][:], p2[1][:]
            if random.random() < crossover_rate:
                size = len(perm1)
                start = random.randint(0, size-1)
                end = random.randint(start, size-1)
                child1 = [None]*size
                child2 = [None]*size
                # copy segment from parent1 to child1
                child1[start:end+1] = perm1[start:end+1]
                # fill remaining from parent2 in order
                pos = end+1
                for x in perm2:
                    if x not in child1:
                        if pos >= size:
                            pos = 0
                        while pos < start:
                            if child1[pos] is None:
                                child1[pos] = x
                                break
                            pos += 1
                        else:
                            while pos <= size-1:
                                if child1[pos] is None:
                                    child1[pos] = x
                                    pos += 1
                                    break
                                pos += 1
                child2[start:end+1] = perm2[start:end+1]
                pos = end+1
                for x in perm1:
                    if x not in child2:
                        if pos >= size:
                            pos = 0
                        while pos < start:
                            if child2[pos] is None:
                                child2[pos] = x
                                break
                            pos += 1
                        else:
                            while pos <= size-1:
                                if child2[pos] is None:
                                    child2[pos] = x
                                    pos += 1
                                    break
                                pos += 1
                perm1, perm2 = child1, child2
            # mutation: swap two random customers
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(perm1)), 2)
                perm1[i], perm1[j] = perm1[j], perm1[i]
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(perm2)), 2)
                perm2[i], perm2[j] = perm2[j], perm2[i]
            # decode
            routes1 = split_permutation(perm1, truck_count)
            routes2 = split_permutation(perm2, truck_count)
            # local search
            routes1, fit1 = local_improve(routes1, max_iter_local // 2, stagnation_limit)
            routes2, fit2 = local_improve(routes2, max_iter_local // 2, stagnation_limit)
            # reconstruct permutations
            perm1 = [c for r in routes1 for c in r[1:-1]]
            perm2 = [c for r in routes2 for c in r[1:-1]]
            new_pop.append((fit1, perm1, routes1))
            new_pop.append((fit2, perm2, routes2))
        # combine and keep best
        combined = population + new_pop
        combined.sort(key=lambda x: x[0])
        population = combined[:pop_size]
        # update best
        if population[0][0] < best_fitness * (1 - 1e-12):
            best_fitness = population[0][0]
            best_routes = population[0][2]
            report_best_vrp(best_routes)
    
    # Ensure exactly truck_count routes
    final_routes = []
    for r in best_routes:
        if len(r) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append([0] + r[1:-1] + [0])
    return final_routes