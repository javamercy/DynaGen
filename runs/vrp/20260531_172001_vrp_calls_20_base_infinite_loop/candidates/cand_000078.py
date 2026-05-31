import numpy as np
import random
import time

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    start_time = time.time()
    max_time = 170

    # --- Helper functions ---
    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def max_route_len(routes):
        return max(route_length(r) for r in routes)

    # Parent's min-max greedy construction with regret tie-breaking (from cand_000055)
    def construct_minmax_regret():
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

    # Decoder: split permutation into equal-sized segments
    def decode(perm):
        customers = perm  # list of 1..n-1 in order
        num_cust = n - 1
        base = num_cust // truck_count
        extra = num_cust % truck_count
        routes = []
        idx = 0
        for i in range(truck_count):
            seg_len = base + (1 if i < extra else 0)
            if seg_len == 0:
                routes.append([0, 0])
            else:
                seg = customers[idx:idx+seg_len]
                routes.append([0] + seg + [0])
                idx += seg_len
        return routes

    # GA operators
    def crossover_ox(p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        remaining = [gene for gene in p2 if gene not in child]
        pos = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[pos]
                pos += 1
        return child

    def mutate_swap(perm):
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
        return perm

    def tournament_selection(pop, fits, k):
        best = random.randrange(len(pop))
        for _ in range(k-1):
            i = random.randrange(len(pop))
            if fits[i] < fits[best]:
                best = i
        return pop[best][:]

    # --- Initialize population ---
    pop_size = min(30, max(10, n * truck_count // 2))
    max_generations = max(100, n * truck_count)
    elite_size = 2

    # Seed with parent construction
    seed_routes = construct_minmax_regret()
    # Convert routes to permutation: concatenate customer order
    seed_perm = []
    for r in seed_routes:
        seed_perm.extend(r[1:-1])
    # Ensure all customers present
    all_cust = set(range(1,n))
    if set(seed_perm) != all_cust:
        seed_perm = list(range(1,n))

    population = [seed_perm]
    while len(population) < pop_size:
        perm = list(range(1, n))
        random.shuffle(perm)
        population.append(perm)

    # Evaluate fitness
    def evaluate(perm):
        routes = decode(perm)
        return max_route_len(routes), routes

    fitness = [evaluate(ind)[0] for ind in population]
    best_routes = None
    best_fit = float('inf')

    for gen in range(max_generations):
        if time.time() - start_time > max_time:
            break

        new_pop = []
        # Elitism
        sorted_idx = sorted(range(len(population)), key=lambda i: fitness[i])
        for i in range(elite_size):
            idx = sorted_idx[i]
            new_pop.append(population[idx][:])
            if fitness[idx] < best_fit:
                best_fit = fitness[idx]
                best_routes = decode(population[idx])
                report_best_vrp(best_routes)

        # Fill rest via crossover and mutation
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitness, 2)
            p2 = tournament_selection(population, fitness, 2)
            child = crossover_ox(p1, p2)
            if random.random() < 0.1:
                child = mutate_swap(child)
            new_pop.append(child)

        population = new_pop
        fitness = [evaluate(ind)[0] for ind in population]

        # Update best if improved
        for i, f in enumerate(fitness):
            if f < best_fit:
                best_fit = f
                best_routes = decode(population[i])
                report_best_vrp(best_routes)

    # If no improvement found, return seed
    if best_routes is None:
        best_routes = seed_routes
    return best_routes