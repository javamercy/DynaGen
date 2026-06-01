import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def compute_route_length(route):
        if len(route) <= 1:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    best_max = float('inf')
    best_routes = None

    def report_best(routes):
        nonlocal best_max, best_routes
        m = max(compute_route_length(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]

    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0.0] * truck_count
        for cust in perm:
            best_max = float('inf')
            best_r = -1
            best_p = -1
            best_total_inc = float('inf')
            for r in range(truck_count):
                route = routes[r]
                for p in range(1, len(route)):
                    prev, nxt = route[p-1], route[p]
                    new_len = lengths[r] - distance_matrix[prev, nxt] + distance_matrix[prev, cust] + distance_matrix[cust, nxt]
                    new_max = new_len
                    for rr in range(truck_count):
                        if rr != r and lengths[rr] > new_max:
                            new_max = lengths[rr]
                    total_inc = new_len - lengths[r]
                    if new_max < best_max or (new_max == best_max and total_inc < best_total_inc):
                        best_max = new_max
                        best_r = r
                        best_p = p
                        best_total_inc = total_inc
            routes[best_r].insert(best_p, cust)
            lengths[best_r] = compute_route_length(routes[best_r])
        return routes, lengths, max(lengths)

    def local_search_best(routes, lengths):
        improved = True
        while improved:
            improved = False
            current_max = max(lengths)
            best_delta = 0
            best_move = None
            # relocate
            for t1 in range(truck_count):
                if len(routes[t1]) <= 2:
                    continue
                for i in range(1, len(routes[t1])-1):
                    cust = routes[t1][i]
                    for t2 in range(truck_count):
                        if t2 == t1:
                            continue
                        for j in range(1, len(routes[t2])):
                            new_route1 = routes[t1][:i] + routes[t1][i+1:]
                            new_len1 = compute_route_length(new_route1)
                            new_route2 = routes[t2][:j] + [cust] + routes[t2][j:]
                            new_len2 = compute_route_length(new_route2)
                            new_max = max(new_len1, new_len2, max(lengths[k] for k in range(truck_count) if k not in (t1, t2)))
                            delta = new_max - current_max
                            if delta < best_delta:
                                best_delta = delta
                                best_move = ('relocate', t1, i, t2, j, cust, new_route1, new_route2, new_len1, new_len2)
            # swap
            for t1 in range(truck_count):
                if len(routes[t1]) <= 2:
                    continue
                for i in range(1, len(routes[t1])-1):
                    cust1 = routes[t1][i]
                    for t2 in range(truck_count):
                        if t2 == t1 or len(routes[t2]) <= 2:
                            continue
                        for j in range(1, len(routes[t2])-1):
                            cust2 = routes[t2][j]
                            new_route1 = routes[t1][:i] + [cust2] + routes[t1][i+1:]
                            new_len1 = compute_route_length(new_route1)
                            new_route2 = routes[t2][:j] + [cust1] + routes[t2][j+1:]
                            new_len2 = compute_route_length(new_route2)
                            new_max = max(new_len1, new_len2, max(lengths[k] for k in range(truck_count) if k not in (t1, t2)))
                            delta = new_max - current_max
                            if delta < best_delta:
                                best_delta = delta
                                best_move = ('swap', t1, i, t2, j, cust1, cust2, new_route1, new_route2, new_len1, new_len2)
            # 2-opt
            for t in range(truck_count):
                if len(routes[t]) <= 3:
                    continue
                for i in range(1, len(routes[t])-2):
                    for j in range(i+1, len(routes[t])-1):
                        new_route = routes[t][:i] + routes[t][i:j+1][::-1] + routes[t][j+1:]
                        new_len = compute_route_length(new_route)
                        new_max = max(new_len, max(lengths[k] for k in range(truck_count) if k != t))
                        delta = new_max - current_max
                        if delta < best_delta:
                            best_delta = delta
                            best_move = ('2opt', t, new_route, new_len)
            if best_delta < 0:
                improved = True
                if best_move[0] == 'relocate':
                    _, t1, i, t2, j, cust, r1, r2, l1, l2 = best_move
                    routes[t1] = r1
                    routes[t2] = r2
                    lengths[t1] = l1
                    lengths[t2] = l2
                elif best_move[0] == 'swap':
                    _, t1, i, t2, j, c1, c2, r1, r2, l1, l2 = best_move
                    routes[t1] = r1
                    routes[t2] = r2
                    lengths[t1] = l1
                    lengths[t2] = l2
                else: # 2opt
                    _, t, new_route, new_len = best_move
                    routes[t] = new_route
                    lengths[t] = new_len
        return routes, lengths

    pop_size = 20
    max_gen = 50
    mutation_prob = 0.1

    population = []
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes, lengths, max_len = decode(perm)
        routes, lengths = local_search_best(routes, lengths)
        report_best(routes)
        max_len = max(lengths)
        population.append((max_len, perm))
    population.sort(key=lambda x: x[0])

    for gen in range(max_gen):
        new_pop = []
        for _ in range(pop_size):
            # tournament selection (size 3)
            idx1 = random.sample(range(pop_size), 3)
            idx2 = random.sample(range(pop_size), 3)
            parent1 = min([population[i] for i in idx1], key=lambda x: x[0])
            parent2 = min([population[i] for i in idx2], key=lambda x: x[0])
            p1, p2 = parent1[1], parent2[1]
            # order crossover
            n_cust = len(customers)
            a = random.randint(0, n_cust-1)
            b = random.randint(0, n_cust-1)
            if a > b:
                a, b = b, a
            child = [None] * n_cust
            child[a:b+1] = p1[a:b+1]
            pos = b+1 if b+1 < n_cust else 0
            for i in range(n_cust):
                val = p2[(b+1+i) % n_cust]
                if val not in child:
                    child[pos] = val
                    pos = (pos + 1) % n_cust
            # fill remaining (should not happen)
            for i in range(n_cust):
                if child[i] is None:
                    child[i] = [c for c in customers if c not in child][0]
            # mutate
            if random.random() < mutation_prob:
                i = random.randint(0, n_cust-1)
                j = random.randint(0, n_cust-1)
                child[i], child[j] = child[j], child[i]
            # evaluate
            routes_child, lengths_child, max_child = decode(child)
            routes_child, lengths_child = local_search_best(routes_child, lengths_child)
            max_child = max(lengths_child)
            report_best(routes_child)
            new_pop.append((max_child, child))
        population = new_pop
        population.sort(key=lambda x: x[0])
    return best_routes