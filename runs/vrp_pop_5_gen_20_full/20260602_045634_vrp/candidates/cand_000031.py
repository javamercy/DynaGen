import numpy as np
import random
import itertools

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    k = truck_count

    # Decoding: regret insertion minimizing max distance, tie-break by current route distance
    def decode(perm):
        routes = [[0, 0] for _ in range(k)]
        dists = [0.0] * k
        for cust in perm:
            # compute best insertion cost and position for each route
            best_inc_r = [float('inf')] * k
            best_pos_r = [-1] * k
            for r in range(k):
                route = routes[r]
                for pos in range(1, len(route)):
                    inc = dists[r] + distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                    if inc < best_inc_r[r]:
                        best_inc_r[r] = inc
                        best_pos_r[r] = pos
            # choose route that minimizes new max distance
            best_route = -1
            best_new_max = float('inf')
            best_curr_dist = float('inf')
            for r in range(k):
                new_dist = dists[r] + best_inc_r[r]
                new_max = max(new_dist, max(dists[:r] + dists[r+1:]))
                if new_max < best_new_max or (new_max == best_new_max and dists[r] < best_curr_dist):
                    best_new_max = new_max
                    best_route = r
                    best_curr_dist = dists[r]
            # insert
            route = routes[best_route]
            pos = best_pos_r[best_route]
            route.insert(pos, cust)
            dists[best_route] += best_inc_r[best_route]
        routes = [list(r) for r in routes]
        maxd = max(dists)
        return routes, maxd

    # ---- Initial population ----
    pop_size = 20
    population = []
    # heuristic: farthest-from-depot first
    farthest = sorted(customers, key=lambda c: distance_matrix[0, c], reverse=True)
    population.append(farthest[:])
    for _ in range(pop_size - 1):
        perm = list(customers)
        random.shuffle(perm)
        population.append(perm)

    def evaluate(perm):
        routes, maxd = decode(perm)
        return maxd, routes

    fitness = []
    best_overall_max = float('inf')
    best_overall_routes = None
    for perm in population:
        maxd, routes = evaluate(perm)
        fitness.append(maxd)
        if maxd < best_overall_max:
            best_overall_max = maxd
            best_overall_routes = routes
            report_best_vrp(best_overall_routes)

    # ---- GA loop ----
    max_generations = 5 * n
    gen = 0
    no_improve = 0

    while gen < max_generations and no_improve < 10:
        gen += 1
        new_pop = []
        # elitism: keep best 2
        elite_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])[:2]
        for idx in elite_indices:
            new_pop.append(population[idx][:])
        # tournament selection
        def tournament(pop, fits):
            a, b = random.sample(range(len(pop)), 2)
            return pop[a] if fits[a] < fits[b] else pop[b]
        # generate offspring
        while len(new_pop) < pop_size:
            p1 = tournament(population, fitness)
            p2 = tournament(population, fitness)
            # order crossover (OX)
            size = len(p1)
            start, end = sorted(random.sample(range(size), 2))
            child = [None] * size
            child[start:end+1] = p2[start:end+1]
            fill = [x for x in p1 if x not in child]
            ptr = 0
            for i in range(size):
                if child[i] is None:
                    child[i] = fill[ptr]
                    ptr += 1
            # swap mutation
            if random.random() < 0.1:
                i, j = random.sample(range(size), 2)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        # evaluate
        population = new_pop
        new_fitness = []
        improved = False
        for i, perm in enumerate(population):
            maxd, routes = evaluate(perm)
            new_fitness.append(maxd)
            if maxd < best_overall_max:
                best_overall_max = maxd
                best_overall_routes = routes
                report_best_vrp(best_overall_routes)
                improved = True
        fitness = new_fitness
        if improved:
            no_improve = 0
        else:
            no_improve += 1

    # ---- Post-processing local search ----
    routes = best_overall_routes
    def route_dist(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total
    def max_dist(routes):
        return max(route_dist(r) for r in routes)
    current_max = max_dist(routes)
    # Relocate (inter-route)
    improved = True
    while improved:
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
                        other_dists = [route_dist(routes[i]) for i in range(k) if i not in (src, dst)]
                        new_max = max(d_src, d_dst, max(other_dists)) if other_dists else max(d_src, d_dst)
                        if new_max < current_max:
                            routes[src] = new_src
                            routes[dst] = new_dst
                            current_max = new_max
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    # 2-opt intra-route
    for r in range(k):
        route = routes[r]
        it = 0
        max_iter = len(route) * 2
        while it < max_iter:
            it += 1
            improved_local = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist(new_route)
                    other_dists = [route_dist(routes[t]) for t in range(k) if t != r]
                    new_max = max(new_dist, max(other_dists)) if other_dists else new_dist
                    if new_max < current_max:
                        routes[r] = new_route
                        route = new_route
                        current_max = new_max
                        improved_local = True
                        report_best_vrp(routes)
                        break
                if improved_local:
                    break
            if not improved_local:
                break

    # Ensure exactly truck_count routes
    while len(routes) < truck_count:
        routes.append([0, 0])
    return routes