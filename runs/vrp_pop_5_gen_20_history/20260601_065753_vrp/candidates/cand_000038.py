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

    # Parameters
    pop_size = min(50, n)
    max_gen = 5 * n
    mutation_prob = 0.1

    # Initialize population with heuristic (nearest neighbor from random start)
    population = []
    best_max = float('inf')
    best_routes = None
    for _ in range(pop_size):
        # Generate permutation via nearest neighbor starting from a random customer
        unvisited = set(customers)
        start = random.choice(customers)
        perm = [start]
        unvisited.remove(start)
        current = start
        while unvisited:
            nearest = min(unvisited, key=lambda c: distance_matrix[current, c])
            perm.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        routes, max_len = decode(perm)
        population.append((max_len, perm))
        report_best_vrp(routes)
    population.sort(key=lambda x: x[0])

    # Steady-state genetic algorithm
    for _ in range(max_gen):
        # Binary tournament selection
        idx1 = random.sample(range(pop_size), 2)
        idx2 = random.sample(range(pop_size), 2)
        parent1 = population[idx1[0]] if population[idx1[0]] <= population[idx1[1]] else population[idx1[1]]
        parent2 = population[idx2[0]] if population[idx2[0]] <= population[idx2[1]] else population[idx2[1]]
        p1 = parent1[1]
        p2 = parent2[1]

        # Order crossover (OX)
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

        # Mutation
        if random.random() < mutation_prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            child[i], child[j] = child[j], child[i]

        # Evaluate child
        routes_child, max_child = decode(child)
        report_best_vrp(routes_child)

        # Replace worst if child is better and not duplicate
        if max_child < population[-1][0]:
            population[-1] = (max_child, child)
            population.sort(key=lambda x: x[0])

    return best_routes