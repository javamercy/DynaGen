import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_length(route):
        if len(route) <= 1:
            return 0.0
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
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
                    prev = route[p-1]
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

    def best_improvement_ls(routes, lengths):
        # best-accept local search: iterate until no improvement (max_iter = 5*n)
        max_iter = 5 * n
        for _ in range(max_iter):
            best_move = None
            best_new_max = max(lengths)
            best_tie = None
            # relocate moves
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for idx1 in range(1, len(route1)-1):
                    cust = route1[idx1]
                    new_route1 = route1[:idx1] + route1[idx1+1:]
                    len1_new = route_length(new_route1)
                    for t2 in range(truck_count):
                        if t1 == t2:
                            continue
                        route2 = routes[t2]
                        for pos in range(1, len(route2)):
                            new_route2 = route2[:pos] + [cust] + route2[pos:]
                            len2_new = route_length(new_route2)
                            new_max = max(len1_new, len2_new)
                            for rr in range(truck_count):
                                if rr not in (t1, t2):
                                    if lengths[rr] > new_max:
                                        new_max = lengths[rr]
                            tie = (t1, idx1, t2, pos)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('relocate', t1, idx1, t2, pos)
                                best_tie = tie
                            elif new_max == best_new_max and (best_tie is None or tie < best_tie):
                                best_new_max = new_max
                                best_move = ('relocate', t1, idx1, t2, pos)
                                best_tie = tie
            # swap moves
            for t1 in range(truck_count):
                route1 = routes[t1]
                if len(route1) <= 2:
                    continue
                for idx1 in range(1, len(route1)-1):
                    cust1 = route1[idx1]
                    for t2 in range(t1+1, truck_count):
                        route2 = routes[t2]
                        if len(route2) <= 2:
                            continue
                        for idx2 in range(1, len(route2)-1):
                            cust2 = route2[idx2]
                            new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                            new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                            len1_new = route_length(new_route1)
                            len2_new = route_length(new_route2)
                            new_max = max(len1_new, len2_new)
                            for rr in range(truck_count):
                                if rr not in (t1, t2):
                                    if lengths[rr] > new_max:
                                        new_max = lengths[rr]
                            tie = (t1, idx1, t2, idx2)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('swap', t1, idx1, t2, idx2)
                                best_tie = tie
                            elif new_max == best_new_max and (best_tie is None or tie < best_tie):
                                best_new_max = new_max
                                best_move = ('swap', t1, idx1, t2, idx2)
                                best_tie = tie
            # 2-opt moves
            for t in range(truck_count):
                route = routes[t]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_len = route_length(new_route)
                        new_max = new_len
                        for rr in range(truck_count):
                            if rr != t:
                                if lengths[rr] > new_max:
                                    new_max = lengths[rr]
                        tie = (t, i, j)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('2opt', t, i, j)
                            best_tie = tie
                        elif new_max == best_new_max and (best_tie is None or tie < best_tie):
                            best_new_max = new_max
                            best_move = ('2opt', t, i, j)
                            best_tie = tie
            if best_move is None or best_new_max >= max(lengths) - 1e-12:
                break
            # apply best move
            if best_move[0] == 'relocate':
                _, t1, idx1, t2, pos = best_move
                cust = routes[t1][idx1]
                del routes[t1][idx1]
                routes[t2].insert(pos, cust)
            elif best_move[0] == 'swap':
                _, t1, idx1, t2, idx2 = best_move
                cust1 = routes[t1][idx1]
                cust2 = routes[t2][idx2]
                routes[t1][idx1] = cust2
                routes[t2][idx2] = cust1
            else: # '2opt'
                _, t, i, j = best_move
                routes[t] = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
            lengths = [route_length(r) for r in routes]
            report_best_vrp(routes)
        return routes, lengths

    pop_size = min(30, n)
    max_gen = 10 * n
    stagnation_limit = max_gen // 5

    population = []
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes, lengths, _ = decode(perm)
        routes, lengths = best_improvement_ls(routes, lengths)
        report_best_vrp(routes)
        population.append((max(lengths), perm))
    population.sort(key=lambda x: x[0])

    no_improve = 0
    for gen in range(max_gen):
        # adaptive mutation probability
        mutation_prob = 0.3 * (1 - gen / max_gen) + 0.2 * (no_improve / stagnation_limit)
        mutation_prob = min(0.5, mutation_prob)
        # linear ranking selection
        pop_sorted = sorted(population, key=lambda x: x[0])
        ranks = list(range(1, pop_size+1))  # 1..pop_size
        total_rank = sum(ranks)
        selection_probs = [r / total_rank for r in ranks]
        idx1 = random.choices(range(pop_size), weights=selection_probs)[0]
        idx2 = random.choices(range(pop_size), weights=selection_probs)[0]
        parent1 = pop_sorted[idx1]
        parent2 = pop_sorted[idx2]
        p1, p2 = parent1[1], parent2[1]

        n_cust = len(customers)
        # Order Crossover (OX)
        a = random.randint(0, n_cust-1)
        b = random.randint(0, n_cust-1)
        if a > b:
            a, b = b, a
        child = [None] * n_cust
        child[a:b+1] = p1[a:b+1]
        pos = (b+1) % n_cust
        for i in range(n_cust):
            idx = (i + (b+1)) % n_cust
            if p2[idx] not in child:
                child[pos] = p2[idx]
                pos = (pos+1) % n_cust
        # fill any missing (should not happen)
        used = set(child)
        if len(used) != n_cust:
            remaining = [c for c in customers if c not in used]
            for i in range(n_cust):
                if child[i] is None:
                    child[i] = remaining.pop()

        # swap mutation with adaptive probability
        if random.random() < mutation_prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            if i != j:
                child[i], child[j] = child[j], child[i]

        routes_child, lengths_child, _ = decode(child)
        routes_child, lengths_child = best_improvement_ls(routes_child, lengths_child)
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
                routes, lengths = best_improvement_ls(routes, lengths)
                population[i] = (max(lengths), perm)
                report_best_vrp(routes)
            population.sort(key=lambda x: x[0])

    return best_routes