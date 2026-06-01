import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def compute_route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        for cust in perm:
            best_max = float('inf')
            best_r = -1
            best_p = -1
            for r in range(truck_count):
                route = routes[r]
                for p in range(1, len(route)):
                    prev = route[p-1]
                    nxt = route[p]
                    new_len = lengths[r] - distance_matrix[prev, nxt] + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != r and lengths[rr] > new_max:
                            new_max = lengths[rr]
                    if new_max < best_max:
                        best_max = new_max
                        best_r = r
                        best_p = p
                    elif new_max == best_max:
                        if r < best_r or (r == best_r and p < best_p):
                            best_max = new_max
                            best_r = r
                            best_p = p
            routes[best_r].insert(best_p, cust)
            lengths[best_r] = compute_route_length(routes[best_r])
        max_len = max(lengths)
        return routes, max_len

    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]

    pop_size = min(100, n)
    max_gen = 10 * n

    population = []
    best_max = float('inf')
    best_routes = None
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes, max_len = decode(perm)
        population.append((max_len, perm))
        report_best_vrp(routes)
    population.sort(key=lambda x: x[0])

    stagnation = 0
    best_fitness = population[0][0]
    stagnation_limit = max_gen // 5

    for gen in range(1, max_gen + 1):
        # Adaptive crossover probability: 0.9 to 0.5 linearly
        crossover_prob = 0.9 - 0.4 * (gen / max_gen)
        # Adaptive mutation probability: base from 0.2 to 0.05, but increase on stagnation
        base_mutation_prob = 0.2 - 0.15 * (gen / max_gen)
        if stagnation >= stagnation_limit:
            mutation_prob = min(base_mutation_prob + 0.1, 0.3)
        else:
            mutation_prob = base_mutation_prob

        # Binary tournament selection
        idx1 = random.sample(range(pop_size), 2)
        idx2 = random.sample(range(pop_size), 2)
        parent1 = population[idx1[0]] if population[idx1[0]] <= population[idx1[1]] else population[idx1[1]]
        parent2 = population[idx2[0]] if population[idx2[0]] <= population[idx2[1]] else population[idx2[1]]
        p1 = parent1[1]
        p2 = parent2[1]

        # PMX crossover
        n_cust = len(customers)
        if random.random() < crossover_prob:
            a = random.randint(0, n_cust-1)
            b = random.randint(0, n_cust-1)
            if a > b:
                a, b = b, a
            child = [None] * n_cust
            # copy segment from parent1
            child[a:b+1] = p1[a:b+1]
            # map for PMX
            mapping = {}
            for i in range(a, b+1):
                mapping[p2[i]] = p1[i]
            # fill positions outside segment with parent2's elements, resolving conflicts
            for i in list(range(0, a)) + list(range(b+1, n_cust)):
                cand = p2[i]
                while cand in child:
                    cand = mapping[cand]
                child[i] = cand
            # fill any remaining None (should not happen)
            used = set(child)
            if len(used) != n_cust:
                remaining = [c for c in customers if c not in used]
                for i in range(n_cust):
                    if child[i] is None:
                        child[i] = remaining.pop()
        else:
            child = p1[:]

        # swap mutation
        if random.random() < mutation_prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            child[i], child[j] = child[j], child[i]

        routes_child, max_child = decode(child)
        report_best_vrp(routes_child)

        # steady-state replacement (replace worst)
        if max_child < population[-1][0]:
            population[-1] = (max_child, child)
            population.sort(key=lambda x: x[0])
            if max_child < best_fitness:
                best_fitness = max_child
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1

    return best_routes