import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        total = 0.0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def max_route_len(routes):
        return max(route_length(r) for r in routes) if routes else float('inf')

    def decode(perm):
        # greedy assignment: assign each customer to the truck with smallest current route length
        routes = [[0, 0] for _ in range(truck_count)]
        route_lens = [0.0] * truck_count
        for cust in perm:
            # find truck with minimum current length, break ties by index
            min_len = route_lens[0]
            min_idx = 0
            for t in range(1, truck_count):
                if route_lens[t] < min_len:
                    min_len = route_lens[t]
                    min_idx = t
            # insert cust just before the final depot
            routes[min_idx].insert(-1, cust)
            # update route length: add edge from previous to cust and cust to depot, subtract previous to depot
            # but easier: recompute after insertion? For speed, approximate by adding distances
            # For exact, we need to recompute series. We'll recompute after all assignments.
            # After all assignments, we compute exact lengths.
        # recompute exact route lengths
        for t in range(truck_count):
            route_lens[t] = route_length(routes[t])
        return routes, max(route_lens)

    # initial population
    pop_size = 20
    max_gens = 50
    pop = []
    fitness = []

    # generate one solution using parent's construction (min-max regret with insertion)
    # We'll reuse the construction from parent but as a standalone function
    def construct_greedy():
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = set(range(1, n))
        while unassigned:
            candidates = []
            for cust in unassigned:
                insert_info = []
                for r_idx, route in enumerate(routes):
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

    # create initial population
    heuristic_routes = construct_greedy()
    heuristic_perm = []
    for r in heuristic_routes:
        heuristic_perm.extend(r[1:-1])  # exclude depots
    pop.append(heuristic_perm)
    fitness.append(max_route_len(heuristic_routes))
    report_best_vrp(heuristic_routes)
    best_routes = [r[:] for r in heuristic_routes]
    best_fitness = fitness[0]

    # random permutations for rest
    customers = list(range(1, n))
    for _ in range(pop_size - 1):
        random.shuffle(customers)
        perm = customers[:]
        pop.append(perm)
        rts, f = decode(perm)
        fitness.append(f)
        if f < best_fitness:
            best_fitness = f
            best_routes = [r[:] for r in rts]
            report_best_vrp(best_routes)

    # genetic algorithm main loop
    no_improve_gen = 0
    for gen in range(max_gens):
        # selection (tournament) and reproduction
        new_pop = []
        new_fitness = []
        # elitism: keep best individual
        best_idx = int(np.argmin(fitness))
        new_pop.append(pop[best_idx][:])
        new_fitness.append(fitness[best_idx])
        # fill rest
        while len(new_pop) < pop_size:
            # tournament selection
            idx1 = random.randint(0, pop_size - 1)
            idx2 = random.randint(0, pop_size - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, pop_size - 1)
            if fitness[idx1] < fitness[idx2]:
                parent1 = pop[idx1][:]
            else:
                parent1 = pop[idx2][:]
            idx1 = random.randint(0, pop_size - 1)
            idx2 = random.randint(0, pop_size - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, pop_size - 1)
            if fitness[idx1] < fitness[idx2]:
                parent2 = pop[idx1][:]
            else:
                parent2 = pop[idx2][:]
            # crossover with probability 0.8
            if random.random() < 0.8:
                # order crossover (OX)
                size = len(parent1)
                a, b = sorted(random.sample(range(size), 2))
                child = [None] * size
                child[a:b+1] = parent1[a:b+1]
                pos = b+1
                for i in range(size):
                    if pos >= size:
                        pos = 0
                    if parent2[i] not in child:
                        child[pos] = parent2[i]
                        pos += 1
                        if pos >= size:
                            pos = 0
                # mutation with probability 0.1
                if random.random() < 0.1:
                    i, j = random.sample(range(size), 2)
                    child[i], child[j] = child[j], child[i]
            else:
                child = parent1[:]
                if random.random() < 0.1:
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
            rts, f = decode(child)
            new_pop.append(child)
            new_fitness.append(f)
            if f < best_fitness:
                best_fitness = f
                best_routes = [r[:] for r in rts]
                report_best_vrp(best_routes)
                no_improve_gen = 0
        pop = new_pop
        fitness = new_fitness
        no_improve_gen += 1
        if no_improve_gen >= 20:
            break
    return best_routes