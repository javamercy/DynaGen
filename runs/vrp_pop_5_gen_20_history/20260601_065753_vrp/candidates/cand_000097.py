import numpy as np
import random


def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_length(route):
        if len(route) <= 1:
            return 0.0
        total = 0.0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i], route[i + 1]]
        return total

    best_max = float('inf')
    best_routes = None

    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(route_length(r) for r in routes)
        if m < best_max - 1e-12:
            best_max = m
            best_routes = [list(r) for r in routes]

    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        for cust in perm:
            best_max_val = float('inf')
            best_inc = float('inf')
            best_r = -1
            best_p = -1
            for r in range(truck_count):
                route = routes[r]
                for p in range(1, len(route)):
                    prev = route[p - 1]
                    nxt = route[p]
                    new_len = lengths[r] - distance_matrix[prev, nxt] + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != r and lengths[rr] > new_max:
                            new_max = lengths[rr]
                    inc = new_len - lengths[r]
                    if new_max < best_max_val or (abs(new_max - best_max_val) < 1e-12 and inc < best_inc):
                        best_max_val = new_max
                        best_inc = inc
                        best_r = r
                        best_p = p
            routes[best_r].insert(best_p, cust)
            lengths[best_r] = route_length(routes[best_r])
        return routes, lengths, max(lengths)

    def local_search(routes, lengths):
        max_iter = 2 * n
        for _ in range(max_iter):
            improved = False
            # try move types: relocate, 2-opt, swap
            move_type = random.randint(0, 2)
            if move_type == 0:
                max_len = max(lengths)
                longest = [i for i, l in enumerate(lengths) if abs(l - max_len) < 1e-12]
                if longest:
                    t1 = random.choice(longest)
                    route1 = routes[t1]
                    if len(route1) > 2:
                        i = random.randint(1, len(route1) - 2)
                        cust = route1[i]
                        t2 = random.randint(0, truck_count - 1)
                        if t2 != t1:
                            j = random.randint(1, len(routes[t2]) - 1)
                            new_route1 = route1[:i] + route1[i + 1:]
                            new_len1 = route_length(new_route1)
                            new_route2 = routes[t2][:j] + [cust] + routes[t2][j:]
                            new_len2 = route_length(new_route2)
                            new_max = max(new_len1, new_len2)
                            for k in range(truck_count):
                                if k not in (t1, t2) and lengths[k] > new_max:
                                    new_max = lengths[k]
                            if new_max < max_len - 1e-12:
                                routes[t1] = new_route1
                                lengths[t1] = new_len1
                                routes[t2] = new_route2
                                lengths[t2] = new_len2
                                improved = True
                                report_best_vrp(routes)
            elif move_type == 1:
                max_len = max(lengths)
                longest = [i for i, l in enumerate(lengths) if abs(l - max_len) < 1e-12]
                if longest:
                    t = random.choice(longest)
                    route = routes[t]
                    if len(route) > 3:
                        i = random.randint(1, len(route) - 3)
                        j = random.randint(i + 1, len(route) - 2)
                        new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        new_len = route_length(new_route)
                        new_max = new_len
                        for k in range(truck_count):
                            if k != t and lengths[k] > new_max:
                                new_max = lengths[k]
                        if new_max < max_len - 1e-12:
                            routes[t] = new_route
                            lengths[t] = new_len
                            improved = True
                            report_best_vrp(routes)
            else:
                # intra-route swap
                max_len = max(lengths)
                longest = [i for i, l in enumerate(lengths) if abs(l - max_len) < 1e-12]
                if longest:
                    t = random.choice(longest)
                    route = routes[t]
                    if len(route) > 3:
                        i = random.randint(1, len(route) - 2)
                        j = random.randint(1, len(route) - 2)
                        if i != j:
                            new_route = route[:]
                            new_route[i], new_route[j] = new_route[j], new_route[i]
                            new_len = route_length(new_route)
                            new_max = new_len
                            for k in range(truck_count):
                                if k != t and lengths[k] > new_max:
                                    new_max = lengths[k]
                            if new_max < max_len - 1e-12:
                                routes[t] = new_route
                                lengths[t] = new_len
                                improved = True
                                report_best_vrp(routes)
            if not improved:
                break
        return routes, lengths

    pop_size = min(30, n)
    max_gen = 10 * n
    stagnation_limit = max_gen // 5

    population = []
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes, lengths, _ = decode(perm)
        routes, lengths = local_search(routes, lengths)
        report_best_vrp(routes)
        population.append((max(lengths), perm))
    population.sort(key=lambda x: x[0])

    no_improve = 0
    for gen in range(max_gen):
        mutation_prob = 0.3 * (1 - gen / max_gen)
        # Linear ranking selection
        pop_sorted = sorted(population, key=lambda x: x[0])
        ranks = list(range(pop_size))
        total_rank = sum(ranks) + pop_size  # ranks from 0 to pop_size-1, add 1 to avoid zero
        selection_probs = [(r + 1) / total_rank for r in range(pop_size)]
        idx1 = random.choices(range(pop_size), weights=selection_probs)[0]
        idx2 = random.choices(range(pop_size), weights=selection_probs)[0]
        # We must ensure parents are distinct? Not strictly required; can be same.
        parent1 = pop_sorted[idx1]
        parent2 = pop_sorted[idx2]
        p1, p2 = parent1[1], parent2[1]

        n_cust = len(customers)
        # Order Crossover (OX)
        a = random.randint(0, n_cust - 1)
        b = random.randint(0, n_cust - 1)
        if a > b:
            a, b = b, a
        child = [None] * n_cust
        child[a:b + 1] = p1[a:b + 1]
        pos = (b + 1) % n_cust
        for i in range(n_cust):
            idx = (i + (b + 1)) % n_cust
            if p2[idx] not in child:
                child[pos] = p2[idx]
                pos = (pos + 1) % n_cust
        # fill any None (should not happen, but safe)
        used = set(child)
        if len(used) != n_cust:
            remaining = [c for c in customers if c not in used]
            for i in range(n_cust):
                if child[i] is None:
                    child[i] = remaining.pop()

        # swap mutation
        if random.random() < mutation_prob:
            i = random.randint(0, n_cust - 1)
            j = random.randint(0, n_cust - 1)
            if i != j:
                child[i], child[j] = child[j], child[i]

        routes_child, lengths_child, _ = decode(child)
        routes_child, lengths_child = local_search(routes_child, lengths_child)
        report_best_vrp(routes_child)

        if max(lengths_child) < population[-1][0] - 1e-12:
            population[-1] = (max(lengths_child), child)
            population.sort(key=lambda x: x[0])
            if max(lengths_child) < best_max - 1e-12:
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        if no_improve >= stagnation_limit:
            no_improve = 0
            for i in range(pop_size // 2, pop_size):
                perm = customers[:]
                random.shuffle(perm)
                routes, lengths, _ = decode(perm)
                routes, lengths = local_search(routes, lengths)
                population[i] = (max(lengths), perm)
                report_best_vrp(routes)
            population.sort(key=lambda x: x[0])

    return best_routes