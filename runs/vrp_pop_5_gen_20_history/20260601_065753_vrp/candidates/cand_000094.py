import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def compute_route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0 for _ in range(truck_count)]
        for cust in perm:
            r = min(range(truck_count), key=lambda i: lengths[i])
            routes[r].insert(-1, cust)
            lengths[r] = compute_route_length(routes[r])
        max_len = max(lengths)
        return routes, max_len

    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]

    pop_size = min(80, n)
    max_gen = 5 * n
    mut_start = 0.2
    mut_end = 0.05

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

    for gen in range(max_gen):
        mutation_prob = mut_start - (gen / max_gen) * (mut_start - mut_end)
        # Binary tournament selection
        idx1 = random.sample(range(pop_size), 2)
        idx2 = random.sample(range(pop_size), 2)
        parent1 = population[idx1[0]] if population[idx1[0]][0] <= population[idx1[1]][0] else population[idx1[1]]
        parent2 = population[idx2[0]] if population[idx2[0]][0] <= population[idx2[1]][0] else population[idx2[1]]
        p1 = parent1[1]
        p2 = parent2[1]

        n_cust = len(customers)
        a = random.randint(0, n_cust-1)
        b = random.randint(0, n_cust-1)
        if a > b:
            a, b = b, a
        child = [None] * n_cust
        child[a:b+1] = p1[a:b+1]
        pos = b+1
        for gene in p2:
            if gene not in child:
                if pos >= n_cust:
                    pos = 0
                child[pos] = gene
                pos += 1

        # Swap mutation with adaptive prob
        if random.random() < mutation_prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            if i != j:
                child[i], child[j] = child[j], child[i]

        routes_child, max_child = decode(child)
        report_best_vrp(routes_child)

        if max_child < population[-1][0]:
            population[-1] = (max_child, child)
            population.sort(key=lambda x: x[0])

    return best_routes