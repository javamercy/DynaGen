import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    max_dist = np.max(distance_matrix)

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def two_opt(route):
        improved = True
        it = 0
        while improved and it < n:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
        return route

    def regret2_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0] * truck_count
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = (distance_matrix[route[pos-1], cust] +
                               distance_matrix[cust, route[pos]] -
                               distance_matrix[route[pos-1], route[pos]])
                        incs.append((inc, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                regret = incs[1][0] - incs[0][0] if len(incs) >= 2 else 0.0
                if regret > best_regret or (regret == best_regret and incs[0][0] < best_inc):
                    best_regret = regret
                    best_inc = incs[0][0]
                    best_cust = cust
                    best_route_idx = incs[0][2]
                    best_pos = incs[0][1]
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        lengths = [route_distance(r) for r in routes]
        return routes, lengths

    def split_permutation(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0] * truck_count
        for cust in perm:
            best_truck = -1
            best_new_max = float('inf')
            best_pos = -1
            best_total = float('inf')
            for t in range(truck_count):
                best_inc = float('inf')
                best_p = -1
                for p in range(1, len(routes[t])):
                    inc = (distance_matrix[routes[t][p-1], cust] +
                           distance_matrix[cust, routes[t][p]] -
                           distance_matrix[routes[t][p-1], routes[t][p]])
                    if inc < best_inc:
                        best_inc = inc
                        best_p = p
                new_len = lengths[t] + best_inc
                other_lengths = [lengths[i] for i in range(truck_count) if i != t]
                new_max = max(new_len, max(other_lengths) if other_lengths else 0)
                new_total = new_len + sum(other_lengths)
                if (new_max < best_new_max or
                    (new_max == best_new_max and new_total < best_total) or
                    (new_max == best_new_max and new_total == best_total and t < best_truck)):
                    best_new_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = best_p
            routes[best_truck].insert(best_pos, cust)
            lengths[best_truck] = route_distance(routes[best_truck])
        return routes, lengths

    def evaluate(perm):
        routes, lengths = split_permutation(perm)
        # apply 2-opt to each route
        for i in range(truck_count):
            if len(routes[i]) > 3:
                routes[i] = two_opt(routes[i])
                lengths[i] = route_distance(routes[i])
        return max(lengths), routes, lengths

    # GA parameters
    pop_size = min(30, n * 2)
    generations = min(50, n * 3)
    elite_count = 2
    crossover_rate = 0.8
    mutation_rate = 0.1

    # Initial population
    population = []
    greedy_routes, _ = regret2_construction()
    greedy_perm = []
    for route in greedy_routes:
        greedy_perm.extend(route[1:-1])
    population.append(greedy_perm)

    while len(population) < pop_size:
        perm = customers[:]
        random.shuffle(perm)
        population.append(perm)

    best_fitness = float('inf')
    best_routes = None

    for perm in population:
        fit, routes, _ = evaluate(perm)
        if fit < best_fitness:
            best_fitness = fit
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    for gen in range(generations):
        # Evaluate population
        evaluated = []
        for perm in population:
            fit, routes, _ = evaluate(perm)
            evaluated.append((fit, routes, perm))
        evaluated.sort(key=lambda x: x[0])

        # Elitism
        new_pop = [list(evaluated[i][2]) for i in range(elite_count)]

        while len(new_pop) < pop_size:
            # Tournament selection (size 3)
            idxs = random.sample(range(pop_size), 3)
            best_idx = min(idxs, key=lambda i: evaluated[i][0])
            parent1 = evaluated[best_idx][2]
            idxs = random.sample(range(pop_size), 3)
            best_idx = min(idxs, key=lambda i: evaluated[i][0])
            parent2 = evaluated[best_idx][2]

            if random.random() < crossover_rate:
                # Order Crossover (OX1)
                m = len(parent1)
                start = random.randint(0, m-1)
                end = random.randint(start, min(start+m-1, m-1))
                child = [None] * m
                child[start:end+1] = parent1[start:end+1]
                ptr = (end + 1) % m
                for elem in parent2:
                    if elem not in child:
                        child[ptr] = elem
                        ptr = (ptr + 1) % m
            else:
                child = list(parent1)

            # Swap mutation
            if random.random() < mutation_rate:
                i = random.randint(0, len(child)-1)
                j = random.randint(0, len(child)-1)
                child[i], child[j] = child[j], child[i]

            new_pop.append(child)

        population = new_pop

        # Update best
        for perm in population:
            fit, routes, _ = evaluate(perm)
            if fit < best_fitness:
                best_fitness = fit
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)

    return best_routes