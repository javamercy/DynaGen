import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    num_cust = n - 1
    k = truck_count

    # ---------- decode: permutation -> routes ----------
    def decode(perm):
        routes = [[0, 0] for _ in range(k)]
        dists = [0.0] * k
        def route_dist(route):
            return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
        for cust in perm:
            best_increase = float('inf')
            best_route = -1
            best_pos = -1
            for r in range(k):
                route = routes[r]
                for pos in range(1, len(route)):
                    new_dist = dists[r] - distance_matrix[route[pos-1], route[pos]] \
                               + distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]]
                    # new_max after insertion
                    new_max = max(new_dist, max(dists[:r] + dists[r+1:]))
                    cur_max = max(dists)
                    increase = new_max - cur_max
                    # tie-break by smaller total distance
                    if increase < best_increase or (increase == best_increase and dists[r] < dists[best_route]):
                        best_increase = increase
                        best_route = r
                        best_pos = pos
            route = routes[best_route]
            route.insert(best_pos, cust)
            dists[best_route] = route_dist(route)
        # ensure exactly k routes
        routes = [list(r) for r in routes]
        return routes, max(dists)

    # ---------- local search ----------
    def local_search(routes):
        def route_dist(route):
            return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
        def max_dist(routes):
            return max(route_dist(r) for r in routes)
        best_max = max_dist(routes)
        improved = True
        # relocate
        max_iter = n * 2
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for src in range(k):
                route_src = routes[src][:]
                for idx in range(1, len(route_src)-1):
                    cust = route_src[idx]
                    new_src = route_src[:idx] + route_src[idx+1:]
                    d_src = route_dist(new_src)
                    for dst in range(k):
                        if dst == src:
                            continue
                        route_dst = routes[dst][:]
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
        # 2-opt internal
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for r in range(k):
                route = routes[r][:]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_dist(new_route)
                        new_max = max(new_dist, max(route_dist(rt) for ri, rt in enumerate(routes) if ri != r))
                        if new_max < best_max:
                            routes[r] = new_route
                            best_max = new_max
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes, best_max

    # ---------- GA run (one restart) ----------
    def run_ga():
        # initial population
        pop_size = 30
        population = []
        # heuristic: farthest-first
        farthest = sorted(customers, key=lambda c: distance_matrix[0, c], reverse=True)
        population.append(farthest)
        # nearest neighbor heuristic (starting from depot)
        nn = []
        unvisited = set(customers)
        cur = 0
        while unvisited:
            next_cust = min(unvisited, key=lambda c: distance_matrix[cur, c])
            nn.append(next_cust)
            unvisited.remove(next_cust)
            cur = next_cust
        population.append(nn)
        # rest random
        for _ in range(pop_size - 2):
            perm = customers[:]
            random.shuffle(perm)
            population.append(perm)
        # evaluate
        fitness = []
        decoded = []
        best_max = float('inf')
        best_routes = None
        for perm in population:
            routes, maxd = decode(perm)
            fitness.append(maxd)
            decoded.append(routes)
            if maxd < best_max:
                best_max = maxd
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
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
                return pop[a] if fits[a] < fits[b] else pop[b]
            while len(new_pop) < pop_size:
                p1 = tournament(population, fitness)
                p2 = tournament(population, fitness)
                # order crossover
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
                # mutation (swap)
                if random.random() < 0.2:
                    i, j = random.sample(range(size), 2)
                    child[i], child[j] = child[j], child[i]
                new_pop.append(child)
            # evaluate new population
            population = new_pop
            new_fitness = []
            new_decoded = []
            improved = False
            for perm in population:
                routes, maxd = decode(perm)
                new_fitness.append(maxd)
                new_decoded.append(routes)
                if maxd < best_max:
                    best_max = maxd
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                    improved = True
            fitness = new_fitness
            decoded = new_decoded
            if improved:
                no_improve = 0
                # local search on best solution
                best_routes, best_max = local_search([list(r) for r in best_routes])
                report_best_vrp(best_routes)
            else:
                no_improve += 1
        return best_routes, best_max

    # run multiple restarts
    best_overall_max = float('inf')
    best_overall_routes = None
    for restart in range(3):
        # re-seed for reproducibility but different restart
        random.seed(restart)
        routes, maxd = run_ga()
        if maxd < best_overall_max:
            best_overall_max = maxd
            best_overall_routes = routes
            report_best_vrp(routes)

    # final local search on overall best
    routes, _ = local_search([list(r) for r in best_overall_routes])
    return routes