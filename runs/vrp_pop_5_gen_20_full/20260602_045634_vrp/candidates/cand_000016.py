import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    k = truck_count

    def route_dist(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    def decode(perm):
        routes = [[0, 0] for _ in range(k)]
        dists = [0.0] * k
        for cust in perm:
            best_increase = float('inf')
            best_route = -1
            best_pos = -1
            for r in range(k):
                route = routes[r]
                for pos in range(1, len(route)):
                    new_d = dists[r] + distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                    cur_max = max(dists)
                    new_max = max(new_d, max(dists[:r] + dists[r+1:]))
                    increase = new_max - cur_max
                    if increase < best_increase or (increase == best_increase and (r < best_route or (r == best_route and pos < best_pos))):
                        best_increase = increase
                        best_route = r
                        best_pos = pos
            route = routes[best_route]
            route.insert(best_pos, cust)
            dists[best_route] = route_dist(route)
        return [list(r) for r in routes], max(dists)

    # initial population
    pop_size = 20
    population = []
    farthest = sorted(customers, key=lambda c: distance_matrix[0, c], reverse=True)
    population.append(farthest)
    for i in range(pop_size-1):
        perm = list(customers)
        random.shuffle(perm)
        population.append(perm)

    def evaluate(perm):
        return decode(perm)

    fitness = []
    best_routes = None
    best_fit = float('inf')
    for perm in population:
        routes, fit = evaluate(perm)
        fitness.append(fit)
        if fit < best_fit:
            best_fit = fit
            best_routes = routes
            report_best_vrp(best_routes)

    # local search on a solution: relocate and 2-opt, returns improved solution
    def local_search(routes):
        cur_routes = [list(r) for r in routes]
        cur_max = max_dist(cur_routes)
        improved = True
        max_iter = n * 2
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            # relocate
            for src in range(k):
                if len(cur_routes[src]) > 2:
                    for idx in range(1, len(cur_routes[src])-1):
                        cust = cur_routes[src][idx]
                        new_src = cur_routes[src][:idx] + cur_routes[src][idx+1:]
                        d_src = route_dist(new_src)
                        for dst in range(k):
                            if dst == src:
                                continue
                            route_dst = cur_routes[dst]
                            for pos in range(1, len(route_dst)):
                                new_dst = route_dst[:pos] + [cust] + route_dst[pos:]
                                d_dst = route_dist(new_dst)
                                others_max = max(route_dist(r) for i, r in enumerate(cur_routes) if i not in (src, dst))
                                new_max = max(d_src, d_dst, others_max)
                                if new_max < cur_max:
                                    cur_routes[src] = new_src
                                    cur_routes[dst] = new_dst
                                    cur_max = new_max
                                    improved = True
                                    report_best_vrp(cur_routes)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                continue
            # 2-opt within each route
            for r in range(k):
                route = cur_routes[r]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_d = route_dist(new_route)
                        others_max = max(route_dist(cur) for idxr, cur in enumerate(cur_routes) if idxr != r)
                        new_max = max(new_d, others_max)
                        if new_max < cur_max:
                            cur_routes[r] = new_route
                            cur_max = new_max
                            improved = True
                            report_best_vrp(cur_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
        return cur_routes

    # GA
    max_gen = 10 * n
    no_improve = 0
    for gen in range(max_gen):
        # selection and crossover
        new_pop = []
        # elitism: keep best 2
        sorted_idx = sorted(range(len(fitness)), key=lambda i: fitness[i])
        for idx in sorted_idx[:2]:
            new_pop.append(population[idx][:])
        while len(new_pop) < pop_size:
            # tournament selection
            a, b = random.sample(range(len(population)), 2)
            p1 = population[a] if fitness[a] < fitness[b] else population[b]
            c, d = random.sample(range(len(population)), 2)
            p2 = population[c] if fitness[c] < fitness[d] else population[d]
            # order crossover (OX)
            size = len(p1)
            start, end = sorted(random.sample(range(size), 2))
            child = [None]*size
            child[start:end+1] = p2[start:end+1]
            fill = [x for x in p1 if x not in child]
            ptr = 0
            for i in range(size):
                if child[i] is None:
                    child[i] = fill[ptr]
                    ptr += 1
            # swap mutation
            if random.random() < 0.2:
                i, j = random.sample(range(size), 2)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        # evaluate new population
        population = new_pop
        new_fitness = []
        new_best = False
        for i, perm in enumerate(population):
            routes, fit = evaluate(perm)
            # apply local search
            routes = local_search(routes)
            fit = max_dist(routes)
            new_fitness.append(fit)
            if fit < best_fit:
                best_fit = fit
                best_routes = routes
                report_best_vrp(best_routes)
                new_best = True
        fitness = new_fitness
        if new_best:
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 20:
            break

    # final post-processing VND
    routes = [list(r) for r in best_routes]
    improved = True
    max_iter = n * 3
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        # relocate
        for src in range(k):
            if len(routes[src]) <= 2:
                continue
            for idx in range(1, len(routes[src])-1):
                cust = routes[src][idx]
                new_src = routes[src][:idx] + routes[src][idx+1:]
                d_src = route_dist(new_src)
                for dst in range(k):
                    if dst == src:
                        continue
                    route_dst = routes[dst]
                    for pos in range(1, len(route_dst)):
                        new_dst = route_dst[:pos] + [cust] + route_dst[pos:]
                        d_dst = route_dist(new_dst)
                        others_max = max(route_dist(r) for i, r in enumerate(routes) if i not in (src, dst))
                        new_max = max(d_src, d_dst, others_max)
                        if new_max < best_fit:
                            routes[src] = new_src
                            routes[dst] = new_dst
                            best_fit = new_max
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 2-opt*
        for r1 in range(k):
            for r2 in range(r1+1, k):
                route1 = routes[r1]
                route2 = routes[r2]
                if len(route1) <= 2 or len(route2) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        new1 = route1[:i] + route2[j:] 
                        new2 = route2[:j] + route1[i:]
                        if new1[-1] != 0:
                            new1.append(0)
                        if new2[-1] != 0:
                            new2.append(0)
                        if new1[0] != 0:
                            new1 = [0] + new1
                        if new2[0] != 0:
                            new2 = [0] + new2
                        # ensure start and end with 0
                        if new1[0] != 0 or new1[-1] != 0:
                            continue
                        if new2[0] != 0 or new2[-1] != 0:
                            continue
                        d1 = route_dist(new1)
                        d2 = route_dist(new2)
                        others_max = max(route_dist(r) for i, r in enumerate(routes) if i not in (r1, r2))
                        new_max = max(d1, d2, others_max)
                        if new_max < best_fit:
                            routes[r1] = new1
                            routes[r2] = new2
                            best_fit = new_max
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 2-opt within each route
        for r in range(k):
            route = routes[r]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_d = route_dist(new_route)
                    others_max = max(route_dist(cur) for idxr, cur in enumerate(routes) if idxr != r)
                    new_max = max(new_d, others_max)
                    if new_max < best_fit:
                        routes[r] = new_route
                        best_fit = new_max
                        improved = True
                        report_best_vrp(routes)
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    # ensure exactly k routes
    while len(routes) < k:
        routes.append([0, 0])
    # ensure start and end with 0
    for i in range(k):
        if routes[i][0] != 0:
            routes[i].insert(0, 0)
        if routes[i][-1] != 0:
            routes[i].append(0)
    return routes