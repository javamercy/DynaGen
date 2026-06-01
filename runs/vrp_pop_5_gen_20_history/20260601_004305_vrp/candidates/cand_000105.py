import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    random.seed(0)
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_dist(route):
        return sum(distance_matrix[route[k]][route[k+1]] for k in range(len(route)-1))

    def compute_max(routes):
        return max(route_dist(r) for r in routes)

    def copy_routes(routes):
        return [list(r) for r in routes]

    def split_permutation(perm):
        seg_len = len(perm) // truck_count
        remainder = len(perm) % truck_count
        routes = []
        start = 0
        for t in range(truck_count):
            extra = 1 if t < remainder else 0
            end = start + seg_len + extra
            segment = perm[start:end]
            if segment:
                routes.append([0] + segment + [0])
            else:
                routes.append([0, 0])
            start = end
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def permutation_from_routes(routes):
        perm = []
        for r in routes:
            for c in r:
                if c != 0:
                    perm.append(c)
        return perm

    def apply_two_opt(routes):
        best_routes = copy_routes(routes)
        best_max = compute_max(best_routes)
        improved = True
        max_iter = 50
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for ri in range(len(best_routes)):
                route = best_routes[ri]
                if len(route) <= 3:
                    continue
                best_local_dist = route_dist(route)
                best_local_route = route[:]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_dist(new_route)
                        if new_dist < best_local_dist - 1e-9:
                            best_local_dist = new_dist
                            best_local_route = new_route
                if best_local_route != route:
                    best_routes[ri] = best_local_route
                    new_max = compute_max(best_routes)
                    if new_max < best_max - 1e-9:
                        best_max = new_max
                        improved = True
        return best_routes

    pop_size = 5
    population = []
    best_global_routes = None
    best_global_max = math.inf
    for _ in range(pop_size):
        perm = list(range(1, n))
        random.shuffle(perm)
        routes = split_permutation(perm)
        routes = apply_two_opt(routes)
        max_dist = compute_max(routes)
        population.append((max_dist, routes))
        if max_dist < best_global_max:
            best_global_max = max_dist
            best_global_routes = copy_routes(routes)
            report_best_vrp(best_global_routes)

    generations = 10
    pm = 0.1

    def tournament_select(pop):
        idx = random.sample(range(len(pop)), 2)
        best_idx = min(idx, key=lambda i: pop[i][0])
        return pop[best_idx][1]

    def ox_crossover(p1, p2):
        n_cust = len(p1)
        start = random.randint(0, n_cust-1)
        end = random.randint(start+1, n_cust)
        child = [None] * n_cust
        for i in range(start, end):
            child[i] = p1[i]
        pos = 0
        for i in range(n_cust):
            if child[i] is None:
                while p2[pos] in child:
                    pos += 1
                child[i] = p2[pos]
        return child

    for gen in range(generations):
        parent1_routes = tournament_select(population)
        parent2_routes = tournament_select(population)
        perm1 = permutation_from_routes(parent1_routes)
        perm2 = permutation_from_routes(parent2_routes)
        child_perm = ox_crossover(perm1, perm2)
        if random.random() < pm:
            i = random.randint(0, m-1)
            j = random.randint(0, m-1)
            child_perm[i], child_perm[j] = child_perm[j], child_perm[i]
        child_routes = split_permutation(child_perm)
        child_routes = apply_two_opt(child_routes)
        child_max = compute_max(child_routes)
        if child_max < best_global_max:
            best_global_max = child_max
            best_global_routes = copy_routes(child_routes)
            report_best_vrp(best_global_routes)
        worst_idx = max(range(pop_size), key=lambda i: population[i][0])
        if child_max < population[worst_idx][0]:
            population[worst_idx] = (child_max, child_routes)

    return best_global_routes