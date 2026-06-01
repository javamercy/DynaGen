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
        # Greedy min-max insertion with tie-breaking by total distance increase
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

    def local_search(routes, lengths):
        max_iter_local = 5 * n
        if max_iter_local == 0:
            return routes, lengths
        improved = True
        iteration = 0
        while improved and iteration < max_iter_local:
            improved = False
            iteration += 1
            # relocate moves: move a customer from longest route to another
            max_len = max(lengths)
            candidates = [i for i, l in enumerate(lengths) if abs(l - max_len) < 1e-12]
            t1 = random.choice(candidates)
            route1 = routes[t1]
            if len(route1) <= 2:
                continue
            i = random.randint(1, len(route1)-2)
            cust = route1[i]
            best_delta = 0
            best_t2 = -1
            best_j = -1
            for t2 in range(truck_count):
                if t2 == t1:
                    continue
                route2 = routes[t2]
                for j in range(1, len(route2)):
                    new_len1 = lengths[t1] - distance_matrix[route1[i-1], route1[i]] - distance_matrix[route1[i], route1[i+1]] + distance_matrix[route1[i-1], route1[i+1]]
                    new_len2 = lengths[t2] + distance_matrix[route2[j-1], cust] + distance_matrix[cust, route2[j]] - distance_matrix[route2[j-1], route2[j]]
                    new_max = max(new_len1, new_len2)
                    for k in range(truck_count):
                        if k != t1 and k != t2:
                            if lengths[k] > new_max:
                                new_max = lengths[k]
                    delta = current_max - new_max
                    if delta > best_delta:
                        best_delta = delta
                        best_t2 = t2
                        best_j = j
            if best_delta > 1e-12:
                # perform move
                routes[best_t2].insert(best_j, cust)
                route1.pop(i)
                lengths[t1] = route_length(route1)
                lengths[best_t2] = route_length(routes[best_t2])
                improved = True
                continue
            # 2-opt on longest route
            max_len = max(lengths)
            candidates = [i for i, l in enumerate(lengths) if abs(l - max_len) < 1e-12]
            t = random.choice(candidates)
            route = routes[t]
            if len(route) <= 3:
                continue
            best_delta = 0
            best_i = -1
            best_j = -1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_len = route_length(new_route)
                    new_max = new_len
                    for k in range(truck_count):
                        if k != t:
                            if lengths[k] > new_max:
                                new_max = lengths[k]
                    delta = current_max - new_max
                    if delta > best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_delta > 1e-12:
                new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                routes[t] = new_route
                lengths[t] = route_length(new_route)
                improved = True
        current_max = max(lengths)
        return routes, lengths

    pop_size = min(30, n)
    max_gen = 10 * n
    stagnation_limit = max_gen // 5

    population = []
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes, lengths, max_len = decode(perm)
        routes, lengths = local_search(routes, lengths)
        max_len = max(lengths)
        report_best_vrp(routes)
        population.append((max_len, perm))
    population.sort(key=lambda x: x[0])

    no_improve = 0
    for gen in range(max_gen):
        mutation_prob = 0.3 * (1 - gen / max_gen)
        # binary tournament
        idx1 = random.sample(range(pop_size), 2)
        parent1 = population[idx1[0]] if population[idx1[0]][0] < population[idx1[1]][0] else population[idx1[1]]
        idx2 = random.sample(range(pop_size), 2)
        parent2 = population[idx2[0]] if population[idx2[0]][0] < population[idx2[1]][0] else population[idx2[1]]
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
        # fill any None (should not happen, but safe)
        used = set(child)
        if len(used) != n_cust:
            remaining = [c for c in customers if c not in used]
            for i in range(n_cust):
                if child[i] is None:
                    child[i] = remaining.pop()

        # swap mutation
        if random.random() < mutation_prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            if i != j:
                child[i], child[j] = child[j], child[i]

        routes_child, lengths_child, max_child = decode(child)
        routes_child, lengths_child = local_search(routes_child, lengths_child)
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
                routes, lengths, max_len = decode(perm)
                routes, lengths = local_search(routes, lengths)
                max_len = max(lengths)
                population[i] = (max_len, perm)
                report_best_vrp(routes)
            population.sort(key=lambda x: x[0])

    return best_routes