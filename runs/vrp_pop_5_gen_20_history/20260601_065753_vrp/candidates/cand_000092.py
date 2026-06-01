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
        max_len = max(lengths)
        return routes, lengths, max_len

    def best_accept_local_search(routes, lengths):
        max_iter = 5 * n
        current_max = max(lengths)
        for _ in range(max_iter):
            best_move = None
            best_new_max = current_max
            best_tie = None
            # Relocate moves
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
                                if rr != t1 and rr != t2:
                                    if lengths[rr] > new_max:
                                        new_max = lengths[rr]
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('relocate', t1, idx1, t2, pos)
                                best_tie = (t1, idx1, t2, pos)
                            elif new_max == best_new_max:
                                tie = (t1, idx1, t2, pos)
                                if best_tie is None or tie < best_tie:
                                    best_new_max = new_max
                                    best_move = ('relocate', t1, idx1, t2, pos)
                                    best_tie = tie
            # Swap moves
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
                                if rr != t1 and rr != t2:
                                    if lengths[rr] > new_max:
                                        new_max = lengths[rr]
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = ('swap', t1, idx1, t2, idx2)
                                best_tie = (t1, idx1, t2, idx2)
                            elif new_max == best_new_max:
                                tie = (t1, idx1, t2, idx2)
                                if best_tie is None or tie < best_tie:
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
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('2opt', t, i, j)
                            best_tie = (t, i, j)
                        elif new_max == best_new_max:
                            tie = (t, i, j)
                            if best_tie is None or tie < best_tie:
                                best_new_max = new_max
                                best_move = ('2opt', t, i, j)
                                best_tie = tie
            if best_move is None or best_new_max >= current_max:
                break
            # Apply best move
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
            else:  # '2opt'
                _, t, i, j = best_move
                routes[t] = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
            lengths = [route_length(r) for r in routes]
            current_max = max(lengths)
            if current_max < best_max - 1e-12:
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
        routes, lengths = best_accept_local_search(routes, lengths)
        report_best_vrp(routes)
        population.append((max(lengths), perm))
    population.sort(key=lambda x: x[0])

    no_improve = 0
    for gen in range(max_gen):
        mutation_prob = 0.3 * (1 - gen / max_gen) + 0.2 * (no_improve / stagnation_limit)
        mutation_prob = min(0.5, mutation_prob)
        # binary tournament selection
        idx1 = random.sample(range(pop_size), 2)
        parent1 = population[idx1[0]] if population[idx1[0]][0] < population[idx1[1]][0] else population[idx1[1]]
        idx2 = random.sample(range(pop_size), 2)
        parent2 = population[idx2[0]] if population[idx2[0]][0] < population[idx2[1]][0] else population[idx2[1]]
        p1, p2 = parent1[1], parent2[1]

        n_cust = len(customers)
        # PMX crossover
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
                val = p2[i]
                while val in child:
                    val = mapping[val]
                child[i] = val
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
        routes_child, lengths_child = best_accept_local_search(routes_child, lengths_child)
        max_child = max(lengths_child)
        report_best_vrp(routes_child)

        if max_child < population[-1][0] - 1e-12:
            population[-1] = (max_child, child)
            population.sort(key=lambda x: x[0])
            if max_child < best_max - 1e-12:
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
                routes, lengths = best_accept_local_search(routes, lengths)
                population[i] = (max(lengths), perm)
                report_best_vrp(routes)
            population.sort(key=lambda x: x[0])

    return best_routes