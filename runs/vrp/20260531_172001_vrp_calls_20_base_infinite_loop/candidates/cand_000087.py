import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def max_route_len(routes):
        return max(route_length(r) for r in routes)

    # Decode permutation into routes using greedy min-max insertion
    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in perm:
            best_max = float('inf')
            best_cost = float('inf')
            best_route = None
            best_pos = None
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_len = route_length(route) + cost
                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    if new_max < best_max or (abs(new_max - best_max) < 1e-12 and cost < best_cost):
                        best_max = new_max
                        best_cost = cost
                        best_route = r_idx
                        best_pos = pos
            routes[best_route].insert(best_pos, cust)
        return routes

    # Helper: generate initial construction from regret construction (parent 086 style)
    def regret_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            candidates = []
            for cust in unassigned:
                insert_info = []
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_info.append((new_max, cost, r_idx, pos))
                insert_info.sort(key=lambda x: (x[0], x[1]))
                best = insert_info[0]
                second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                regret = second[0] - best[0]
                candidates.append((best[0], regret, best[1], best[2], best[3], cust))
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
            chosen = candidates[0]
            _, _, _, r_idx, pos, cust = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        return routes

    # Extract permutation from routes: order of customers as they appear in routes (route0 then route1...)
    def routes_to_perm(routes):
        perm = []
        for r in routes:
            perm.extend(r[1:-1])
        return perm

    # Order Crossover (OX1)
    def crossover(parent1, parent2):
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b+1] = parent1[a:b+1]
        remaining = [x for x in parent2 if x not in child[a:b+1]]
        pos = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[pos]
                pos += 1
        return child

    # Mutation: swap two positions
    def mutate(perm):
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]

    # Parameters bounded by instance size
    pop_size = min(20, 10 + n // 10)
    max_gens = min(50, 20 + n // 5)
    mutation_rate = 0.2
    elitism_size = max(1, pop_size // 10)

    # Initial population
    population = []
    # Seed with permutation from regret construction
    init_routes = regret_construction()
    init_perm = routes_to_perm(init_routes)
    population.append(init_perm)
    best_routes = init_routes
    best_max = max_route_len(init_routes)
    report_best_vrp(init_routes)
    # Fill with random permutations
    while len(population) < pop_size:
        perm = list(range(1, n))
        random.shuffle(perm)
        population.append(perm)

    for gen in range(max_gens):
        # Evaluate
        fitness = []
        for perm in population:
            routes = decode(perm)
            cur_max = max_route_len(routes)
            fitness.append((cur_max, routes))
            if cur_max < best_max:
                best_max = cur_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
        # Sort by fitness (lower better)
        sorted_pop = [p for _, p in sorted(zip(fitness, population), key=lambda x: x[0][0])]
        new_population = sorted_pop[:elitism_size]  # elitism
        # Binary tournament selection
        while len(new_population) < pop_size:
            idx1, idx2 = random.sample(range(pop_size), 2)
            p1 = population[idx1]
            p2 = population[idx2]
            f1 = fitness[idx1][0]
            f2 = fitness[idx2][0]
            parent = p1 if f1 < f2 else p2
            # Crossover with another random parent
            other = random.choice(population)
            if random.random() < 0.8:
                child = crossover(parent, other)
            else:
                child = parent[:]
            # Mutation
            if random.random() < mutation_rate:
                mutate(child)
            new_population.append(child)
        population = new_population

    # Final evaluation to ensure best is returned
    if best_routes is None:
        best_routes = decode(population[0])
    return best_routes