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
        # greedy min-max insertion with tie-breaking by total distance increase
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

    def local_search_vnd(routes, lengths, current_best):
        # Variable Neighborhood Descent: relocate, swap, 2-opt
        # Keep applying until no improvement or max iterations
        max_iter = 10 * n  # bound
        iter_count = 0
        improved = True
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # relocate: try moving a customer from the longest route to another route
            longest_route_idx = max(range(truck_count), key=lambda i: lengths[i])
            route_long = routes[longest_route_idx]
            if len(route_long) > 2:
                for i in range(1, len(route_long)-1):
                    cust = route_long[i]
                    for t2 in range(truck_count):
                        if t2 == longest_route_idx:
                            continue
                        route2 = routes[t2]
                        for j in range(1, len(route2)):
                            new_route1 = route_long[:i] + route_long[i+1:]
                            new_len1 = route_length(new_route1)
                            new_route2 = route2[:j] + [cust] + route2[j:]
                            new_len2 = route_length(new_route2)
                            new_max = max(new_len1, new_len2, max(lengths[k] for k in range(truck_count) if k != longest_route_idx and k != t2))
                            if new_max < max(lengths) - 1e-12:
                                # apply move
                                routes[longest_route_idx] = new_route1
                                lengths[longest_route_idx] = new_len1
                                routes[t2] = new_route2
                                lengths[t2] = new_len2
                                improved = True
                                best_max_local = max(lengths)
                                if best_max_local < best_max - 1e-12:
                                    report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                continue
            # swap: swap customers between two routes (prefer longest route involved)
            for r1 in range(truck_count):
                if len(routes[r1]) <= 2:
                    continue
                for r2 in range(r1+1, truck_count):
                    if len(routes[r2]) <= 2:
                        continue
                    for i in range(1, len(routes[r1])-1):
                        for j in range(1, len(routes[r2])-1):
                            cust1 = routes[r1][i]
                            cust2 = routes[r2][j]
                            new_route1 = routes[r1][:i] + [cust2] + routes[r1][i+1:]
                            new_len1 = route_length(new_route1)
                            new_route2 = routes[r2][:j] + [cust1] + routes[r2][j+1:]
                            new_len2 = route_length(new_route2)
                            new_max = max(new_len1, new_len2, max(lengths[k] for k in range(truck_count) if k != r1 and k != r2))
                            if new_max < max(lengths) - 1e-12:
                                routes[r1] = new_route1
                                lengths[r1] = new_len1
                                routes[r2] = new_route2
                                lengths[r2] = new_len2
                                improved = True
                                best_max_local = max(lengths)
                                if best_max_local < best_max - 1e-12:
                                    report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt: intra-route on longest route
            longest_route_idx = max(range(truck_count), key=lambda i: lengths[i])
            route_l = routes[longest_route_idx]
            if len(route_l) <= 3:
                continue
            for i in range(1, len(route_l)-2):
                for j in range(i+1, len(route_l)-1):
                    new_route = route_l[:i] + route_l[i:j+1][::-1] + route_l[j+1:]
                    new_len = route_length(new_route)
                    new_max = new_len
                    for k in range(truck_count):
                        if k != longest_route_idx:
                            new_max = max(new_max, lengths[k])
                    if new_max < max(lengths) - 1e-12:
                        routes[longest_route_idx] = new_route
                        lengths[longest_route_idx] = new_len
                        improved = True
                        best_max_local = max(lengths)
                        if best_max_local < best_max - 1e-12:
                            report_best_vrp(routes)
                        break
                if improved:
                    break
        return routes, lengths

    pop_size = min(30, n)
    max_gen = 10 * n
    stagnation_limit = max_gen // 5

    population = []
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        routes, lengths, max_len = decode(perm)
        routes, lengths = local_search_vnd(routes, lengths, best_max if best_max != float('inf') else max_len)
        max_len = max(lengths)
        report_best_vrp(routes)
        population.append((max_len, perm))
    population.sort(key=lambda x: x[0])

    no_improve = 0
    for gen in range(max_gen):
        mutation_prob = 0.3 * (1 - gen / max_gen)
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

        # swap mutation
        if random.random() < mutation_prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            if i != j:
                child[i], child[j] = child[j], child[i]

        routes_child, lengths_child, max_child = decode(child)
        routes_child, lengths_child = local_search_vnd(routes_child, lengths_child, best_max)
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
                routes, lengths = local_search_vnd(routes, lengths, best_max)
                max_len = max(lengths)
                population[i] = (max_len, perm)
                report_best_vrp(routes)
            population.sort(key=lambda x: x[0])

    return best_routes