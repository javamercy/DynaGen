import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def compute_route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in perm:
            best_truck = 0
            best_new_max = float('inf')
            for t in range(truck_count):
                new_route = routes[t][:-1] + [cust, 0]
                new_len = compute_route_length(new_route)
                new_max = new_len
                for other in range(truck_count):
                    if other != t:
                        other_len = compute_route_length(routes[other])
                        if other_len > new_max:
                            new_max = other_len
                if new_max < best_new_max or (new_max == best_new_max and t < best_truck):
                    best_new_max = new_max
                    best_truck = t
            routes[best_truck].insert(-1, cust)
        max_len = max(compute_route_length(r) for r in routes)
        return routes, max_len

    best_max = float('inf')
    best_routes = None

    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]

    pop_size = min(50, n)
    max_gen = 5 * n
    mutation_prob = 0.1

    population = []
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes, max_len = decode(perm)
        population.append((max_len, perm))
        report_best_vrp(routes)
    population.sort(key=lambda x: x[0])

    for _ in range(max_gen):
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
        mapping = {}
        for i in range(a, b+1):
            mapping[p1[i]] = p2[i]
        for i in range(n_cust):
            if i < a or i > b:
                gene = p2[i]
                while gene in mapping:
                    gene = mapping[gene]
                child[i] = gene
        used = set(child)
        for i in range(n_cust):
            if child[i] is None:
                child[i] = (set(customers) - used).pop()

        if random.random() < mutation_prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            child[i], child[j] = child[j], child[i]

        routes_child, max_child = decode(child)
        report_best_vrp(routes_child)

        if max_child < population[-1][0]:
            population[-1] = (max_child, child)
            population.sort(key=lambda x: x[0])

    return best_routes