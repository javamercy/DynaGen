import numpy as np
import random
import itertools

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    num_customers = n - 1
    k = truck_count
    
    # --- Decoding function: given permutation, build routes by greedy insertion minimizing max increase ---
    def decode(perm):
        routes = [[0, 0] for _ in range(k)]
        dists = [0.0] * k
        # helper to compute route distance
        def route_dist(route):
            total = 0
            for i in range(len(route)-1):
                total += distance_matrix[route[i], route[i+1]]
            return total
        for cust in perm:
            best_increase = float('inf')
            best_route = -1
            best_pos = -1
            for r in range(k):
                route = routes[r]
                # try inserting at each position before final 0
                for pos in range(1, len(route)):
                    new_dist = dists[r] \
                        + distance_matrix[route[pos-1], cust] \
                        + distance_matrix[cust, route[pos]] \
                        - distance_matrix[route[pos-1], route[pos]]
                    new_max = max(new_dist, max(dists[:r] + dists[r+1:]))
                    increase = new_max - max(dists)
                    # tie-break by smaller route index and then position (any)
                    if (increase < best_increase) or (increase == best_increase and r < best_route):
                        best_increase = increase
                        best_route = r
                        best_pos = pos
            # insert
            route = routes[best_route]
            route.insert(best_pos, cust)
            dists[best_route] = route_dist(route)
        return [list(r) for r in routes], max(dists)
    
    # --- Initial population ---
    pop_size = 20
    population = []
    # Heuristic individual: farthest-first ordering
    # start with cust farthest from depot, then nearest neighbor? But we'll just add random
    for _ in range(pop_size):
        perm = list(customers)
        random.shuffle(perm)
        population.append(perm)
    # add one heuristic: sort by distance from depot descending
    farthest = sorted(customers, key=lambda c: distance_matrix[0, c], reverse=True)
    population[0] = farthest  # replace first
    
    # evaluate
    def evaluate(perm):
        routes, maxd = decode(perm)
        return maxd, routes
    
    fitness = []
    decoded_routes = []
    best_overall_max = float('inf')
    best_overall_routes = None
    for perm in population:
        maxd, routes = evaluate(perm)
        fitness.append(maxd)
        decoded_routes.append(routes)
        if maxd < best_overall_max:
            best_overall_max = maxd
            best_overall_routes = routes
            report_best_vrp(best_overall_routes)
    
    # GA parameters
    max_generations = 10 * n
    gen = 0
    no_improve = 0
    
    while gen < max_generations and no_improve < 20:
        gen += 1
        new_pop = []
        # elitism: keep best 2
        elite_idx = sorted(range(len(fitness)), key=lambda i: fitness[i])[:2]
        for idx in elite_idx:
            new_pop.append(population[idx][:])
        # tournament selection
        def tournament(pop, fits):
            a, b = random.sample(range(len(pop)), 2)
            if fits[a] < fits[b]:
                return pop[a]
            else:
                return pop[b]
        # generate offspring
        while len(new_pop) < pop_size:
            p1 = tournament(population, fitness)
            p2 = tournament(population, fitness)
            # order crossover (OX)
            size = len(p1)
            start, end = sorted(random.sample(range(size), 2))
            child = [None]*size
            child[start:end+1] = p2[start:end+1]
            # fill from p1 in order
            fill = [x for x in p1 if x not in child]
            ptr = 0
            for i in range(size):
                if child[i] is None:
                    child[i] = fill[ptr]
                    ptr += 1
            # mutation: swap two random positions
            if random.random() < 0.2:
                i, j = random.sample(range(size), 2)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        # evaluate new population
        population = new_pop
        new_fitness = []
        new_decoded = []
        improved = False
        for i, perm in enumerate(population):
            maxd, routes = evaluate(perm)
            new_fitness.append(maxd)
            new_decoded.append(routes)
            if maxd < best_overall_max:
                best_overall_max = maxd
                best_overall_routes = routes
                report_best_vrp(best_overall_routes)
                improved = True
        fitness = new_fitness
        decoded_routes = new_decoded
        if improved:
            no_improve = 0
        else:
            no_improve += 1
    
    # --- Post-processing: local search on best solution (from parents) ---
    routes = best_overall_routes
    # Helper functions
    def route_dist(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    def max_dist(routes):
        return max(route_dist(r) for r in routes)
    best_max = max_dist(routes)
    # Relocate
    for _ in range(n * 2):
        improved = False
        for src in range(k):
            route_src = routes[src]
            for idx in range(1, len(route_src)-1):
                cust = route_src[idx]
                new_src = route_src[:idx] + route_src[idx+1:]
                d_src = route_dist(new_src)
                for dst in range(k):
                    if dst == src:
                        continue
                    route_dst = routes[dst]
                    for pos in range(1, len(route_dst)):
                        new_dst = route_dst[:pos] + [cust] + route_dst[pos:]
                        d_dst = route_dist(new_dst)
                        new_max = max(d_src, d_dst, max(route_dist(r) for i, r in enumerate(routes) if i not in (src, dst)))
                        if new_max < best_max:
                            routes[src] = new_src
                            routes[dst] = new_dst
                            best_max = new_max
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    # 2-opt internal
    for r in range(k):
        route = routes[r]
        improved = True
        max_iter = len(route) * 2
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    new_max = max(new_dist, max(route_dist(rt) for rti, rt in enumerate(routes) if rti != r))
                    if new_max < best_max:
                        routes[r] = new_route
                        best_max = new_max
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break
    # Ensure exactly truck_count routes (should already)
    while len(routes) < truck_count:
        routes.append([0, 0])
    # Consolidate: remove duplicates? Should not happen.
    return routes