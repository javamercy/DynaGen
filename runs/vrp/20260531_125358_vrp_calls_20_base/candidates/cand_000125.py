import numpy as np
import random
from collections import deque

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    max_dist = np.max(distance_matrix)

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def two_opt(route, max_iter=5):
        route = route[:]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
        return route

    def balance_routes(routes, lengths):
        improved = True
        balance_iter = 0
        max_balance_iter = n
        while improved and balance_iter < max_balance_iter:
            improved = False
            balance_iter += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_pos = None
            best_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max_route)
                min_route = routes[min_idx]
                best_insertion_len = float('inf')
                best_insert_pos = -1
                for p in range(1, len(min_route)):
                    new_min_route = min_route[:p] + [cust] + min_route[p:]
                    l = route_distance(new_min_route)
                    if l < best_insertion_len:
                        best_insertion_len = l
                        best_insert_pos = p
                new_min_route = min_route[:best_insert_pos] + [cust] + min_route[best_insert_pos:]
                new_min_len = route_distance(new_min_route)
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_cust = cust
                    best_pos = best_insert_pos
            if best_cust is not None and best_reduction > 0:
                cust = best_cust
                new_max = [node for node in max_route if node != cust]
                min_route = routes[min_idx]
                new_min = min_route[:best_pos] + [cust] + min_route[best_pos:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_distance(new_max)
                lengths[min_idx] = route_distance(new_min)
                improved = True
        return routes, lengths

    def regret_insertion_with_noise(k=3):
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        noise = random.uniform(0, 0.1 * max_dist)
                        incs.append((inc + noise, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                if len(incs) >= k:
                    regret = sum(incs[i][0] - incs[0][0] for i in range(1, k))
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret or (regret == best_regret and inc < best_inc):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes, [route_distance(r) for r in routes]

    def worst_ruin_recreate(routes, lengths, fraction=0.15):
        n_cust = n - 1
        num_remove = max(1, int(n_cust * fraction))
        savings = []
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)-1):
                cust = route[pos]
                prev = route[pos-1]
                nxt = route[pos+1]
                saving = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                savings.append((saving, cust, r_idx, pos))
        savings.sort(reverse=True)
        to_remove = [item[1] for item in savings[:num_remove]]
        new_routes = [[0, 0] for _ in range(truck_count)]
        for r_idx, route in enumerate(routes):
            new_routes[r_idx] = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
        unvisited = set(to_remove)
        # Reinsert using regret-3 with noise
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route_idx = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(new_routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        noise = random.uniform(0, 0.1 * max_dist)
                        incs.append((inc + noise, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                if len(incs) >= 3:
                    regret = incs[1][0] - incs[0][0] + incs[2][0] - incs[0][0]
                else:
                    regret = 0.0
                inc = incs[0][0]
                pos = incs[0][1]
                r_idx = incs[0][2]
                if regret > best_regret or (regret == best_regret and inc < best_inc):
                    best_regret = regret
                    best_inc = inc
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos
            new_routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        for r_idx in range(truck_count):
            if len(new_routes[r_idx]) > 2:
                new_routes[r_idx] = two_opt(new_routes[r_idx], max_iter=5)
        new_lengths = [route_distance(r) for r in new_routes]
        new_routes, new_lengths = balance_routes(new_routes, new_lengths)
        return new_routes, new_lengths

    def split_permutation(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0] * truck_count
        for cust in perm:
            best_truck = -1
            best_new_max = float('inf')
            best_pos = -1
            best_total = float('inf')
            for t in range(truck_count):
                best_inc = float('inf')
                best_p = -1
                for p in range(1, len(routes[t])):
                    inc = distance_matrix[routes[t][p-1], cust] + distance_matrix[cust, routes[t][p]] - distance_matrix[routes[t][p-1], routes[t][p]]
                    if inc < best_inc:
                        best_inc = inc
                        best_p = p
                new_len = lengths[t] + best_inc
                other_lengths = [lengths[i] for i in range(truck_count) if i != t]
                new_max = max(new_len, max(other_lengths) if other_lengths else 0)
                new_total = new_len + sum(other_lengths)
                if new_max < best_new_max or (new_max == best_new_max and new_total < best_total) or (new_max == best_new_max and new_total == best_total and t < best_truck):
                    best_new_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = best_p
            routes[best_truck].insert(best_pos, cust)
            lengths[best_truck] = route_distance(routes[best_truck])
        return routes, lengths

    def local_search(routes, lengths):
        max_iter = n * truck_count
        for _ in range(max_iter):
            best_move = None
            best_new_max = max(lengths)
            best_total = sum(lengths)
            # relocate
            for cust in range(1, n):
                src_idx = None
                src_pos = None
                for i, route in enumerate(routes):
                    if cust in route:
                        src_idx = i
                        src_pos = route.index(cust)
                        break
                if src_idx is None:
                    continue
                new_src = routes[src_idx][:src_pos] + routes[src_idx][src_pos+1:]
                src_len = route_distance(new_src)
                for dst_idx in range(truck_count):
                    if dst_idx == src_idx:
                        continue
                    if len(routes[dst_idx]) <= 2:
                        continue
                    for ins_pos in range(1, len(routes[dst_idx])):
                        new_dst = routes[dst_idx][:ins_pos] + [cust] + routes[dst_idx][ins_pos:]
                        new_lengths = lengths[:]
                        new_lengths[src_idx] = src_len
                        new_lengths[dst_idx] = route_distance(new_dst)
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_total):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('relocate', src_idx, src_pos, dst_idx, ins_pos, new_src, new_dst)
            # swap
            for i in range(truck_count):
                if len(routes[i]) <= 2:
                    continue
                for ip in range(1, len(routes[i])-1):
                    cust_i = routes[i][ip]
                    for j in range(i+1, truck_count):
                        if len(routes[j]) <= 2:
                            continue
                        for jp in range(1, len(routes[j])-1):
                            cust_j = routes[j][jp]
                            new_i = routes[i][:ip] + [cust_j] + routes[i][ip+1:]
                            new_j = routes[j][:jp] + [cust_i] + routes[j][jp+1:]
                            new_lengths = lengths[:]
                            new_lengths[i] = route_distance(new_i)
                            new_lengths[j] = route_distance(new_j)
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if new_max < best_new_max or (new_max == best_new_max and new_total < best_total):
                                best_new_max = new_max
                                best_total = new_total
                                best_move = ('swap', i, ip, j, jp, new_i, new_j)
            # 2-opt
            for i in range(truck_count):
                if len(routes[i]) <= 3:
                    continue
                for a in range(1, len(routes[i])-2):
                    for b in range(a+1, len(routes[i])-1):
                        new_route = routes[i][:a] + routes[i][a:b+1][::-1] + routes[i][b+1:]
                        new_len = route_distance(new_route)
                        if new_len >= lengths[i]:
                            continue
                        new_lengths = lengths[:]
                        new_lengths[i] = new_len
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_total):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('2opt', i, a, b, new_route)
            if best_move is not None and best_new_max < max(lengths):
                if best_move[0] == 'relocate':
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                elif best_move[0] == 'swap':
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                else:
                    routes[best_move[1]] = best_move[4]
                lengths = [route_distance(r) for r in routes]
            else:
                # balance and break? Only break if no improvement
                routes, lengths = balance_routes(routes, lengths)
                break
        return routes, lengths

    def evaluate(perm):
        routes, lengths = split_permutation(perm)
        routes, lengths = local_search(routes, lengths)
        routes, lengths = balance_routes(routes, lengths)
        return max(lengths), routes

    # GA parameters
    pop_size = min(50, n * 2)
    generations = min(100, n * 5)
    crossover_rate = 0.8
    mutation_rate = 0.1
    elite_count = 2
    # Initial population
    population = []
    # one regret-3 with noise
    greedy_routes, _ = regret_insertion_with_noise(3)
    greedy_perm = []
    for route in greedy_routes:
        greedy_perm.extend(route[1:-1])
    population.append(greedy_perm)
    # random permutations
    while len(population) < pop_size:
        perm = list(customers)
        random.shuffle(perm)
        population.append(perm)

    best_fitness = float('inf')
    best_routes = None
    for perm in population:
        fit, routes = evaluate(perm)
        if fit < best_fitness:
            best_fitness = fit
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    stagnation = 0
    ruin_fraction = 0.15
    for gen in range(generations):
        # Evaluate population
        fit_routes = []
        for idx, perm in enumerate(population):
            fit, routes = evaluate(perm)
            fit_routes.append((fit, routes, perm))
        fit_routes.sort(key=lambda x: x[0])
        # Elitism
        new_pop = [list(p) for _, _, p in fit_routes[:elite_count]]
        # generate offspring
        while len(new_pop) < pop_size:
            # tournament selection
            idx1 = random.randint(0, len(population)-1)
            idx2 = random.randint(0, len(population)-1)
            idx3 = random.randint(0, len(population)-1)
            best_idx = min([idx1, idx2, idx3], key=lambda i: fit_routes[i][0])
            parent1 = fit_routes[best_idx][2]
            idx1 = random.randint(0, len(population)-1)
            idx2 = random.randint(0, len(population)-1)
            idx3 = random.randint(0, len(population)-1)
            best_idx = min([idx1, idx2, idx3], key=lambda i: fit_routes[i][0])
            parent2 = fit_routes[best_idx][2]
            if random.random() < crossover_rate:
                # order crossover
                m = len(parent1)
                start = random.randint(0, m-1)
                end = random.randint(start+1, m)
                child1 = [None] * m
                child1[start:end] = parent1[start:end]
                pos = end
                for gene in parent2:
                    if gene not in child1:
                        child1[pos % m] = gene
                        pos += 1
                child2 = [None] * m
                child2[start:end] = parent2[start:end]
                pos = end
                for gene in parent1:
                    if gene not in child2:
                        child2[pos % m] = gene
                        pos += 1
            else:
                child1, child2 = list(parent1), list(parent2)
            # mutation: swap mutation
            if random.random() < mutation_rate:
                i = random.randint(0, len(child1)-1)
                j = random.randint(0, len(child1)-1)
                child1[i], child1[j] = child1[j], child1[i]
            if random.random() < mutation_rate:
                i = random.randint(0, len(child2)-1)
                j = random.randint(0, len(child2)-1)
                child2[i], child2[j] = child2[j], child2[i]
            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)
        # apply adaptive ruin-recreate to a subset of population (macro-mutation) based on stagnation
        if stagnation > 3:
            ruin_fraction = min(0.3, 0.15 + 0.05 * stagnation)
            # apply to worst solution
            worst_idx = max(range(len(new_pop)), key=lambda i: fit_routes[i][0] if i < len(fit_routes) else float('inf'))
            worst_perm = new_pop[worst_idx]
            worst_routes, worst_lengths = split_permutation(worst_perm)
            worst_routes, worst_lengths = worst_ruin_recreate(worst_routes, worst_lengths, fraction=ruin_fraction)
            # convert back to permutation
            new_perm = []
            for route in worst_routes:
                new_perm.extend(route[1:-1])
            new_pop[worst_idx] = new_perm
            stagnation = 0
        else:
            stagnation += 1 if gen > 0 and fit_routes[0][0] == best_fitness else 0
        population = new_pop
        # Evaluate and update best
        for perm in population:
            fit, routes = evaluate(perm)
            if fit < best_fitness:
                best_fitness = fit
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
    return best_routes