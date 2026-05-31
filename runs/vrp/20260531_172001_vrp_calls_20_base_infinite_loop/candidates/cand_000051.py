import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def max_route_len(routes):
        return max(route_length(r) for r in routes)

    # Decode permutation into routes using greedy insertion that minimizes max route length
    def decode(permutation):
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in permutation:
            best_max = float('inf')
            best_r_idx = 0
            best_pos = 1
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_len = route_length(route) + cost
                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    if new_max < best_max:
                        best_max = new_max
                        best_r_idx = r_idx
                        best_pos = pos
            routes[best_r_idx].insert(best_pos, cust)
        return routes

    def fitness(permutation):
        routes = decode(permutation)
        return max_route_len(routes)

    # Generate random permutation (excluding depot 0)
    def random_perm():
        perm = list(range(1, n))
        random.shuffle(perm)
        return perm

    # Order crossover (OX)
    def order_crossover(p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        pos = (b+1) % size
        for gene in p2:
            if gene not in child:
                child[pos] = gene
                pos = (pos+1) % size
        return [child[i] for i in range(size)]

    # Swap mutation
    def swap_mutate(perm, rate=0.1):
        for i in range(len(perm)):
            if random.random() < rate:
                j = random.randint(0, len(perm)-1)
                perm[i], perm[j] = perm[j], perm[i]
        return perm

    # Population parameters
    pop_size = 20
    max_gen = max(10, n)  # bounded by instance size
    pop = [random_perm() for _ in range(pop_size)]
    fit_list = [fitness(perm) for perm in pop]
    best_perm = min(pop, key=lambda p: fitness(p))
    best_fit = fitness(best_perm)
    best_routes = decode(best_perm)
    report_best_vrp(best_routes)

    for gen in range(max_gen):
        # Binary tournament selection
        new_pop = []
        # Elitism: keep best solution
        elite = min(pop, key=lambda p: fitness(p))
        new_pop.append(elite)
        while len(new_pop) < pop_size:
            i1 = random.randint(0, pop_size-1)
            i2 = random.randint(0, pop_size-1)
            parent1 = pop[i1] if fit_list[i1] < fit_list[i2] else pop[i2]
            i1 = random.randint(0, pop_size-1)
            i2 = random.randint(0, pop_size-1)
            parent2 = pop[i1] if fit_list[i1] < fit_list[i2] else pop[i2]
            child = order_crossover(parent1, parent2)
            child = swap_mutate(child)
            new_pop.append(child)
        # Evaluate new population
        pop = new_pop
        fit_list = [fitness(perm) for perm in pop]
        # Update best
        gen_best_perm = min(pop, key=lambda p: fitness(p))
        gen_best_fit = fitness(gen_best_perm)
        if gen_best_fit < best_fit - 1e-12:
            best_fit = gen_best_fit
            best_perm = gen_best_perm[:]
            best_routes = decode(best_perm)
            report_best_vrp(best_routes)

    # Optionally apply a simple local search to the best solution (2-opt intra route) for polish
    routes = best_routes
    improved = True
    while improved:
        improved = False
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for k in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    if route_length(new_route) < route_length(route) - 1e-12:
                        routes[r_idx] = new_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            new_max = max_route_len(routes)
            if new_max < best_fit - 1e-12:
                best_fit = new_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)

    return best_routes