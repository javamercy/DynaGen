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
        return max(route_length(r) for r in routes) if routes else float('inf')

    # Decode a permutation (excluding depot) into routes using min-max greedy with regret
    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in perm:
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
            second = insert_info[1] if len(insert_info) > 1 else (best[0]+1e9, best[1]+1e9, -1, -1)
            regret = second[0] - best[0]
            candidates.append((best[0], regret, best[1], best[2], best[3], cust))
        # In the parent code, they had outer loop; but here we process one cust at a time
        # Actually we need to embed the candidate generation inside the loop
        # Let's rewrite correctly
        pass

    # Correct decode function:
    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in perm:
            candidates = []
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_len = route_length(route) + cost
                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    candidates.append((new_max, cost, r_idx, pos))
            candidates.sort(key=lambda x: (x[0], x[1]))
            best = candidates[0]
            second = candidates[1] if len(candidates) > 1 else (best[0]+1e9, best[1]+1e9, -1, -1)
            regret = second[0] - best[0]
            # Actually we need to select the insertion that minimizes new_max with regret tie-break
            # But since we process one cust at a time, we just take the best candidate for this cust?
            # In the parent, they had many customers, but here we have one customer per iteration.
            # So we select the best (lowest new_max, then cost) for this cust.
            # To incorporate regret, we would need to evaluate all customers simultaneously, but we are inserting sequentially.
            # So we'll simply pick the best candidate for the current customer (greedy).
            # This matches the construction heuristic if we process customers in the order of the permutation.
            best_candidate = min(candidates, key=lambda x: (x[0], x[1]))
            _, _, r_idx, pos = best_candidate
            routes[r_idx].insert(pos, cust)
        return routes

    # Order crossover (OX) on permutation lists (customers only, no depot)
    def crossover(p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        ptr = (b+1) % size
        for gene in p2:
            if gene not in child:
                child[ptr] = gene
                ptr = (ptr+1) % size
        return child

    def mutate(perm):
        if random.random() < 0.1:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    # Generate initial population using decode with random permutations
    pop_size = min(20, n)
    num_generations = n * 2
    population = []
    best_routes = None
    best_max = float('inf')

    for _ in range(pop_size):
        perm = list(range(1, n))
        random.shuffle(perm)
        routes = decode(perm)
        cur_max = max_route_len(routes)
        if cur_max < best_max:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(routes)
        population.append((routes, cur_max, perm))

    for gen in range(num_generations):
        # Elitism: keep best 2
        population.sort(key=lambda x: x[1])
        new_pop = [population[0], population[1]]
        # Generate offspring
        while len(new_pop) < pop_size:
            # Tournament selection (size 3)
            tourn = random.sample(population, 3)
            tourn.sort(key=lambda x: x[1])
            parent1 = tourn[0][2]
            tourn2 = random.sample(population, 3)
            tourn2.sort(key=lambda x: x[1])
            parent2 = tourn2[0][2]
            # Crossover and mutation
            child_perm = crossover(parent1, parent2)
            child_perm = mutate(child_perm)
            child_routes = decode(child_perm)
            child_max = max_route_len(child_routes)
            if child_max < best_max:
                best_max = child_max
                best_routes = [r[:] for r in child_routes]
                report_best_vrp(child_routes)
            new_pop.append((child_routes, child_max, child_perm))
        population = new_pop

    if best_routes is None:
        # Fallback: return routes from best in population
        population.sort(key=lambda x: x[1])
        best_routes = population[0][0]
    return best_routes