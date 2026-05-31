import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    max_dist = np.max(distance_matrix)

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def two_opt(route):
        improved = True
        it = 0
        while improved and it < n:
            improved = False
            it += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
        return route

    def regret2_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0] * truck_count
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
                        inc = (distance_matrix[route[pos-1], cust] +
                               distance_matrix[cust, route[pos]] -
                               distance_matrix[route[pos-1], route[pos]])
                        incs.append((inc, pos, r_idx))
                incs.sort(key=lambda x: x[0])
                regret = incs[1][0] - incs[0][0] if len(incs) >= 2 else 0.0
                if regret > best_regret or (regret == best_regret and incs[0][0] < best_inc):
                    best_regret = regret
                    best_inc = incs[0][0]
                    best_cust = cust
                    best_route_idx = incs[0][2]
                    best_pos = incs[0][1]
            routes[best_route_idx].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        lengths = [route_distance(r) for r in routes]
        return routes, lengths

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
                    inc = (distance_matrix[routes[t][p-1], cust] +
                           distance_matrix[cust, routes[t][p]] -
                           distance_matrix[routes[t][p-1], routes[t][p]])
                    if inc < best_inc:
                        best_inc = inc
                        best_p = p
                new_len = lengths[t] + best_inc
                other_lengths = [lengths[i] for i in range(truck_count) if i != t]
                new_max = max(new_len, max(other_lengths) if other_lengths else 0)
                new_total = new_len + sum(other_lengths)
                if (new_max < best_new_max or
                    (new_max == best_new_max and new_total < best_total) or
                    (new_max == best_new_max and new_total == best_total and t < best_truck)):
                    best_new_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = best_p
            routes[best_truck].insert(best_pos, cust)
            lengths[best_truck] = route_distance(routes[best_truck])
        return routes, lengths

    def steepest_descent_local_search(routes, lengths):
        improved = True
        max_iter = 10 * n
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            best_delta = 0
            best_move = None
            # Inter-route relocate
            for src_idx in range(truck_count):
                for pos in range(1, len(routes[src_idx])-1):
                    cust = routes[src_idx][pos]
                    new_src = routes[src_idx][:pos] + routes[src_idx][pos+1:]
                    src_len = route_distance(new_src)
                    for dst_idx in range(truck_count):
                        if dst_idx == src_idx:
                            continue
                        for ins_pos in range(1, len(routes[dst_idx])):
                            new_dst = routes[dst_idx][:ins_pos] + [cust] + routes[dst_idx][ins_pos:]
                            dst_len = route_distance(new_dst)
                            new_lengths = lengths[:]
                            new_lengths[src_idx] = src_len
                            new_lengths[dst_idx] = dst_len
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            old_max = max(lengths)
                            old_total = sum(lengths)
                            delta = (new_max - old_max, new_total - old_total)
                            if delta[0] < 0 or (delta[0] == 0 and delta[1] < 0):
                                if delta < best_delta or (delta == best_delta and (src_idx < best_move[0] or (src_idx == best_move[0] and pos < best_move[1]))):
                                    best_delta = delta
                                    best_move = ('relocate', src_idx, pos, dst_idx, ins_pos, new_src, new_dst)
            # Inter-route swap
            for i_idx in range(truck_count):
                for i_pos in range(1, len(routes[i_idx])-1):
                    cust_i = routes[i_idx][i_pos]
                    for j_idx in range(i_idx+1, truck_count):
                        for j_pos in range(1, len(routes[j_idx])-1):
                            cust_j = routes[j_idx][j_pos]
                            new_i = routes[i_idx][:i_pos] + [cust_j] + routes[i_idx][i_pos+1:]
                            new_j = routes[j_idx][:j_pos] + [cust_i] + routes[j_idx][j_pos+1:]
                            new_lengths = lengths[:]
                            new_lengths[i_idx] = route_distance(new_i)
                            new_lengths[j_idx] = route_distance(new_j)
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            old_max = max(lengths)
                            old_total = sum(lengths)
                            delta = (new_max - old_max, new_total - old_total)
                            if delta[0] < 0 or (delta[0] == 0 and delta[1] < 0):
                                if delta < best_delta or (delta == best_delta and (i_idx < best_move[0] or (i_idx == best_move[0] and i_pos < best_move[1]))):
                                    best_delta = delta
                                    best_move = ('swap', i_idx, i_pos, j_idx, j_pos, new_i, new_j)
            if best_move is not None:
                if best_move[0] == 'relocate':
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                else:
                    routes[best_move[1]] = best_move[5]
                    routes[best_move[3]] = best_move[6]
                lengths = [route_distance(r) for r in routes]
                improved = True
        # Intra-route 2-opt
        for idx in range(truck_count):
            if len(routes[idx]) > 3:
                new_route = two_opt(routes[idx])
                if route_distance(new_route) < route_distance(routes[idx]):
                    routes[idx] = new_route
                    lengths[idx] = route_distance(new_route)
        # Balance heuristic
        for _ in range(5 * n):
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if lengths[max_idx] == lengths[min_idx] or len(routes[max_idx]) <= 2:
                break
            best_cust = None
            best_insert_pos = None
            best_reduction = 0
            max_route = routes[max_idx]
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max)
                min_route = routes[min_idx]
                best_inc = float('inf')
                best_p = -1
                for p in range(1, len(min_route)):
                    inc = (distance_matrix[min_route[p-1], cust] +
                           distance_matrix[cust, min_route[p]] -
                           distance_matrix[min_route[p-1], min_route[p]])
                    if inc < best_inc:
                        best_inc = inc
                        best_p = p
                new_min = min_route[:best_p] + [cust] + min_route[best_p:]
                new_min_len = route_distance(new_min)
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_global_max = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_global_max = max(lengths)
                reduction = old_global_max - new_global_max
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_cust = cust
                    best_insert_pos = best_p
            if best_cust is not None and best_reduction > 0:
                routes[max_idx] = [node for node in routes[max_idx] if node != best_cust]
                routes[min_idx] = routes[min_idx][:best_insert_pos] + [best_cust] + routes[min_idx][best_insert_pos:]
                lengths = [route_distance(r) for r in routes]
        return routes, lengths

    def evaluate(perm):
        routes, lengths = split_permutation(perm)
        routes, lengths = steepest_descent_local_search(routes, lengths)
        return max(lengths), routes, lengths

    # GA parameters
    pop_size = min(40, n * 2)
    generations = min(80, n * 3)
    elite_count = 2
    crossover_rate = 0.8
    mutation_rate = 0.1

    # Initial population
    population = []
    greedy_routes, _ = regret2_construction()
    greedy_perm = []
    for route in greedy_routes:
        greedy_perm.extend(route[1:-1])
    population.append(greedy_perm)
    while len(population) < pop_size:
        perm = customers[:]
        random.shuffle(perm)
        population.append(perm)

    best_fitness = float('inf')
    best_routes = None
    for perm in population:
        fit, routes, _ = evaluate(perm)
        if fit < best_fitness:
            best_fitness = fit
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    stagnation = 0
    for gen in range(generations):
        evaluated = []
        for perm in population:
            fit, routes, _ = evaluate(perm)
            evaluated.append((fit, routes, perm))
        evaluated.sort(key=lambda x: x[0])
        new_pop = [list(evaluated[i][2]) for i in range(elite_count)]
        while len(new_pop) < pop_size:
            idx1 = random.randint(0, pop_size-1)
            idx2 = random.randint(0, pop_size-1)
            idx3 = random.randint(0, pop_size-1)
            best_idx = min([idx1, idx2, idx3], key=lambda i: evaluated[i][0])
            parent1 = evaluated[best_idx][2]
            idx1 = random.randint(0, pop_size-1)
            idx2 = random.randint(0, pop_size-1)
            idx3 = random.randint(0, pop_size-1)
            best_idx = min([idx1, idx2, idx3], key=lambda i: evaluated[i][0])
            parent2 = evaluated[best_idx][2]
            if random.random() < crossover_rate:
                m = len(parent1)
                start = random.randint(0, m-1)
                end = random.randint(start, m-1)
                child = [None] * m
                child[start:end+1] = parent1[start:end+1]
                ptr = (end + 1) % m
                for gene in parent2:
                    if gene not in child:
                        child[ptr] = gene
                        ptr = (ptr + 1) % m
            else:
                child = list(parent1)
            if random.random() < mutation_rate:
                i = random.randint(0, len(child)-1)
                j = random.randint(0, len(child)-1)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        population = new_pop
        # Update best
        for perm in population:
            fit, routes, _ = evaluate(perm)
            if fit < best_fitness:
                best_fitness = fit
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
        # Adaptive restart
        if gen > 0 and evaluated[0][0] >= best_fitness:
            stagnation += 1
        else:
            stagnation = 0
        if stagnation >= 2:
            # Replace worst third of population with random permutations
            num_replace = max(1, pop_size // 3)
            for i in range(-num_replace, 0):
                perm = customers[:]
                random.shuffle(perm)
                population[i] = perm
            stagnation = 0
    return best_routes