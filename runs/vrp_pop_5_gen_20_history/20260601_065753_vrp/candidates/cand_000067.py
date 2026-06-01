import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def compute_route_length(route):
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
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]

    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        for cust in perm:
            best_max_candidate = float('inf')
            best_r = -1
            best_p = -1
            best_total_inc = float('inf')
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
                    total_inc = new_len - lengths[r]
                    if new_max < best_max_candidate or (new_max == best_max_candidate and total_inc < best_total_inc):
                        best_max_candidate = new_max
                        best_r = r
                        best_p = p
                        best_total_inc = total_inc
            routes[best_r].insert(best_p, cust)
            lengths[best_r] = compute_route_length(routes[best_r])
        max_len = max(lengths)
        return routes, lengths, max_len

    def local_search(routes, lengths, current_best):
        max_iter_local = 10 * (n + truck_count)
        T0 = current_best * 0.1
        T_end = 0.001
        if max_iter_local > 0:
            cooling = (T_end / T0) ** (1.0 / max_iter_local)
        else:
            cooling = 1.0
        T = T0
        current_max = max(lengths)
        for _ in range(max_iter_local):
            move_type = random.randint(0, 2)
            best_move = None
            best_new_max = float('inf')
            best_tie = None
            if move_type == 0:  # relocate
                t1 = random.randint(0, truck_count-1)
                if len(routes[t1]) <= 2:
                    continue
                i = random.randint(1, len(routes[t1])-2)
                cust = routes[t1][i]
                t2 = random.randint(0, truck_count-1)
                if t2 == t1:
                    continue
                j = random.randint(1, len(routes[t2])-1)
                new_route1 = routes[t1][:i] + routes[t1][i+1:]
                new_len1 = compute_route_length(new_route1)
                new_route2 = routes[t2][:j] + [cust] + routes[t2][j:]
                new_len2 = compute_route_length(new_route2)
                new_max = new_len1
                for k in range(truck_count):
                    if k == t1:
                        new_max = max(new_max, new_len1)
                    elif k == t2:
                        new_max = max(new_max, new_len2)
                    else:
                        new_max = max(new_max, lengths[k])
                tie = (new_max, 0, t1, i, t2, j)
                if best_tie is None or tie < best_tie:
                    best_new_max = new_max
                    best_move = ('relocate', t1, i, t2, j, cust)
                    best_tie = tie
            elif move_type == 1:  # swap
                t1 = random.randint(0, truck_count-1)
                if len(routes[t1]) <= 2:
                    continue
                i = random.randint(1, len(routes[t1])-2)
                cust1 = routes[t1][i]
                t2 = random.randint(0, truck_count-1)
                if t2 == t1:
                    continue
                if len(routes[t2]) <= 2:
                    continue
                j = random.randint(1, len(routes[t2])-2)
                cust2 = routes[t2][j]
                new_route1 = routes[t1][:i] + [cust2] + routes[t1][i+1:]
                new_route2 = routes[t2][:j] + [cust1] + routes[t2][j+1:]
                new_len1 = compute_route_length(new_route1)
                new_len2 = compute_route_length(new_route2)
                new_max = new_len1
                for k in range(truck_count):
                    if k == t1:
                        new_max = max(new_max, new_len1)
                    elif k == t2:
                        new_max = max(new_max, new_len2)
                    else:
                        new_max = max(new_max, lengths[k])
                tie = (new_max, 1, t1, i, t2, j)
                if best_tie is None or tie < best_tie:
                    best_new_max = new_max
                    best_move = ('swap', t1, i, t2, j, cust1, cust2)
                    best_tie = tie
            else:  # 2-opt
                t = random.randint(0, truck_count-1)
                if len(routes[t]) <= 3:
                    continue
                i = random.randint(1, len(routes[t])-3)
                j = random.randint(i+1, len(routes[t])-2)
                new_route = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
                new_len = compute_route_length(new_route)
                new_max = new_len
                for k in range(truck_count):
                    if k != t:
                        new_max = max(new_max, lengths[k])
                tie = (new_max, 2, t, i, j)
                if best_tie is None or tie < best_tie:
                    best_new_max = new_max
                    best_move = ('2opt', t, i, j, new_route)
                    best_tie = tie

            if best_move is None:
                continue

            delta = best_new_max - current_max
            if delta < 0 or random.random() < math.exp(-delta / T):
                if best_move[0] == 'relocate':
                    _, t1, i, t2, j, cust = best_move
                    routes[t1].pop(i)
                    routes[t2].insert(j, cust)
                    lengths[t1] = compute_route_length(routes[t1])
                    lengths[t2] = compute_route_length(routes[t2])
                elif best_move[0] == 'swap':
                    _, t1, i, t2, j, cust1, cust2 = best_move
                    routes[t1][i] = cust2
                    routes[t2][j] = cust1
                    lengths[t1] = compute_route_length(routes[t1])
                    lengths[t2] = compute_route_length(routes[t2])
                else:
                    _, t, i, j, new_route = best_move
                    routes[t] = new_route
                    lengths[t] = new_len
                current_max = max(lengths)
                if current_max < best_max:
                    report_best_vrp(routes)
            T *= cooling
        return routes, lengths

    pop_size = min(100, n)
    max_gen = 5 * n
    stagnation_limit = max_gen // 5

    population = []
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes, lengths, max_len = decode(perm)
        routes, lengths = local_search(routes, lengths, best_max if best_max != float('inf') else max_len)
        report_best_vrp(routes)
        max_len = max(lengths)
        population.append((max_len, perm))
    population.sort(key=lambda x: x[0])

    no_improve_gen = 0
    for gen in range(1, max_gen + 1):
        mutation_prob = 0.3 * (1 - gen / max_gen)
        idx1 = random.sample(range(pop_size), 3)
        idx2 = random.sample(range(pop_size), 3)
        parent1 = min([population[i] for i in idx1], key=lambda x: x[0])
        parent2 = min([population[i] for i in idx2], key=lambda x: x[0])
        p1, p2 = parent1[1], parent2[1]

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

        if random.random() < mutation_prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            child[i], child[j] = child[j], child[i]

        routes_child, lengths_child, max_child = decode(child)
        routes_child, lengths_child = local_search(routes_child, lengths_child, best_max)
        max_child = max(lengths_child)
        report_best_vrp(routes_child)

        if max_child < population[-1][0]:
            population[-1] = (max_child, child)
            population.sort(key=lambda x: x[0])
            if max_child < best_max:
                no_improve_gen = 0
            else:
                no_improve_gen += 1
        else:
            no_improve_gen += 1

        if no_improve_gen >= stagnation_limit:
            no_improve_gen = 0
            for i in range(pop_size // 2, pop_size):
                perm = customers[:]
                random.shuffle(perm)
                routes, lengths, max_len = decode(perm)
                routes, lengths = local_search(routes, lengths, best_max)
                max_len = max(lengths)
                population[i] = (max_len, perm)
                report_best_vrp(routes)
            population.sort(key=lambda x: x[0])

    return best_routes