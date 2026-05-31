import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    max_dist = np.max(distance_matrix)

    def route_distance(route):
        if len(route) < 2:
            return 0.0
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
        max_balance_iter = n
        it = 0
        while improved and it < max_balance_iter:
            improved = False
            it += 1
            max_idx = max(range(truck_count), key=lambda i: lengths[i])
            min_idx = min(range(truck_count), key=lambda i: lengths[i])
            if max_idx == min_idx or lengths[max_idx] == lengths[min_idx]:
                break
            max_route = routes[max_idx]
            best_cust = None
            best_insert_pos = None
            best_reduction = 0
            for pos in range(1, len(max_route)-1):
                cust = max_route[pos]
                new_max_route = max_route[:pos] + max_route[pos+1:]
                new_max_len = route_distance(new_max_route)
                min_route = routes[min_idx]
                best_insertion_len = float('inf')
                best_pos = -1
                for p in range(1, len(min_route)):
                    new_min_route = min_route[:p] + [cust] + min_route[p:]
                    l = route_distance(new_min_route)
                    if l < best_insertion_len:
                        best_insertion_len = l
                        best_pos = p
                new_min_route = min_route[:best_pos] + [cust] + min_route[best_pos:]
                new_min_len = route_distance(new_min_route)
                other_lengths = [lengths[i] for i in range(truck_count) if i not in (max_idx, min_idx)]
                new_max_global = max(new_max_len, new_min_len, max(other_lengths) if other_lengths else 0)
                old_max_global = max(lengths)
                reduction = old_max_global - new_max_global
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_cust = cust
                    best_insert_pos = best_pos
            if best_cust is not None and best_reduction > 0:
                cust = best_cust
                new_max = [node for node in max_route if node != cust]
                min_route = routes[min_idx]
                new_min = min_route[:best_insert_pos] + [cust] + min_route[best_insert_pos:]
                routes[max_idx] = new_max
                routes[min_idx] = new_min
                lengths[max_idx] = route_distance(new_max)
                lengths[min_idx] = route_distance(new_min)
                improved = True
        return routes, lengths

    def regret_insertion_construction(k=3):
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
                        noise = random.uniform(0, 0.05 * max_dist)
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

    def vnd(routes, lengths):
        improved = True
        max_cycles = 10
        cycle = 0
        while improved and cycle < max_cycles:
            improved = False
            cycle += 1
            # Inter-route relocate
            best_move = None
            best_new_max = max(lengths)
            best_total = sum(lengths)
            for cust in range(1, n):
                src_idx = None
                src_pos = None
                for r_idx, route in enumerate(routes):
                    if cust in route:
                        src_idx = r_idx
                        src_pos = route.index(cust)
                        break
                if src_idx is None:
                    continue
                new_src_route = routes[src_idx][:src_pos] + routes[src_idx][src_pos+1:]
                src_len = route_distance(new_src_route)
                for dst_idx in range(truck_count):
                    if dst_idx == src_idx:
                        continue
                    dst_route = routes[dst_idx]
                    if len(dst_route) <= 2:
                        continue
                    for ins_pos in range(1, len(dst_route)):
                        new_dst_route = dst_route[:ins_pos] + [cust] + dst_route[ins_pos:]
                        new_lengths = lengths[:]
                        new_lengths[src_idx] = src_len
                        new_lengths[dst_idx] = route_distance(new_dst_route)
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if (new_max < best_new_max or
                            (new_max == best_new_max and new_total < best_total) or
                            (new_max == best_new_max and new_total == best_total and src_idx < dst_idx)):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('relocate', src_idx, src_pos, dst_idx, ins_pos, new_src_route, new_dst_route)
            if best_move is not None and best_new_max < max(lengths):
                routes[best_move[1]] = best_move[5]
                routes[best_move[3]] = best_move[6]
                lengths = [route_distance(r) for r in routes]
                improved = True
                continue
            # Inter-route swap
            best_move = None
            best_new_max = max(lengths)
            best_total = sum(lengths)
            for i_idx in range(truck_count):
                i_route = routes[i_idx]
                if len(i_route) <= 2:
                    continue
                for i_pos in range(1, len(i_route)-1):
                    cust_i = i_route[i_pos]
                    for j_idx in range(i_idx+1, truck_count):
                        j_route = routes[j_idx]
                        if len(j_route) <= 2:
                            continue
                        for j_pos in range(1, len(j_route)-1):
                            cust_j = j_route[j_pos]
                            new_i_route = i_route[:i_pos] + [cust_j] + i_route[i_pos+1:]
                            new_j_route = j_route[:j_pos] + [cust_i] + j_route[j_pos+1:]
                            new_lengths = lengths[:]
                            new_lengths[i_idx] = route_distance(new_i_route)
                            new_lengths[j_idx] = route_distance(new_j_route)
                            new_max = max(new_lengths)
                            new_total = sum(new_lengths)
                            if (new_max < best_new_max or
                                (new_max == best_new_max and new_total < best_total) or
                                (new_max == best_new_max and new_total == best_total and i_idx < j_idx)):
                                best_new_max = new_max
                                best_total = new_total
                                best_move = ('swap', i_idx, i_pos, j_idx, j_pos, new_i_route, new_j_route)
            if best_move is not None and best_new_max < max(lengths):
                routes[best_move[1]] = best_move[5]
                routes[best_move[3]] = best_move[6]
                lengths = [route_distance(r) for r in routes]
                improved = True
                continue
            # Intra-route 2-opt
            best_move = None
            best_new_max = max(lengths)
            best_total = sum(lengths)
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_len = route_distance(new_route)
                        if new_len >= lengths[r_idx]:
                            continue
                        new_lengths = lengths[:]
                        new_lengths[r_idx] = new_len
                        new_max = max(new_lengths)
                        new_total = sum(new_lengths)
                        if (new_max < best_new_max or
                            (new_max == best_new_max and new_total < best_total) or
                            (new_max == best_new_max and new_total == best_total and r_idx < 0)):
                            best_new_max = new_max
                            best_total = new_total
                            best_move = ('2opt', r_idx, i, j, new_route)
            if best_move is not None and best_new_max < max(lengths):
                routes[best_move[1]] = best_move[4]
                lengths = [route_distance(r) for r in routes]
                improved = True
        return routes, lengths

    def worst_ruin_recreate(routes, lengths, fraction=0.1):
        n_cust = n - 1
        num_remove = max(1, int(n_cust * fraction))
        savings = []
        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)-1):
                cust = route[pos]
                prev = route[pos-1]
                next_ = route[pos+1]
                saving = distance_matrix[prev, cust] + distance_matrix[cust, next_] - distance_matrix[prev, next_]
                savings.append((saving, cust, r_idx, pos))
        savings.sort(reverse=True)
        to_remove = [item[1] for item in savings[:num_remove]]
        new_routes = [[0, 0] for _ in range(truck_count)]
        for r_idx, route in enumerate(routes):
            new_routes[r_idx] = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
        unvisited = set(to_remove)
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
                        noise = random.uniform(0, 0.05 * max_dist)
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

    def pmx_crossover(parent1, parent2):
        m = len(parent1)
        start = random.randint(0, m-2)
        end = random.randint(start+1, m-1)
        child1 = [None] * m
        child2 = [None] * m
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]
        mapping = {}
        for i in range(start, end+1):
            mapping[parent1[i]] = parent2[i]
        for i in range(start, end+1):
            mapping[parent2[i]] = parent1[i]
        for i in range(m):
            if i < start or i > end:
                val = parent2[i]
                while val in mapping:
                    val = mapping[val]
                child1[i] = val
                val = parent1[i]
                while val in mapping:
                    val = mapping[val]
                child2[i] = val
        return child1, child2

    def evaluate(perm):
        routes, lengths = split_permutation(perm)
        routes, lengths = vnd(routes, lengths)
        routes, lengths = balance_routes(routes, lengths)
        return max(lengths), routes

    # GA parameters (from cand_000154)
    pop_size = min(40, n * 2)
    generations = min(80, n * 3)
    crossover_rate = 0.85
    mutation_rate = 0.15
    elite_count = 2
    # Initial population
    population = []
    greedy_routes, _ = regret_insertion_construction(3)
    greedy_perm = []
    for route in greedy_routes:
        greedy_perm.extend(route[1:-1])
    population.append(greedy_perm)
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
        # Generate offspring
        while len(new_pop) < pop_size:
            # ternary tournament selection
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
                child1, child2 = pmx_crossover(parent1, parent2)
            else:
                child1, child2 = list(parent1), list(parent2)
            # scramble mutation
            if random.random() < mutation_rate:
                i = random.randint(0, len(child1)-1)
                j = random.randint(i, len(child1)-1)
                segment = child1[i:j+1]
                random.shuffle(segment)
                child1[i:j+1] = segment
            if random.random() < mutation_rate:
                i = random.randint(0, len(child2)-1)
                j = random.randint(i, len(child2)-1)
                segment = child2[i:j+1]
                random.shuffle(segment)
                child2[i:j+1] = segment
            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)
        # Adaptive ruin-recreate (from cand_000154, applied to worst two)
        if stagnation > 2:
            ruin_fraction = min(0.35, 0.15 + 0.1 * stagnation)
            # Apply to worst two solutions in current population (before replacement)
            # Use fit_routes to get worst permutations
            for i in range(1, 3):
                if i >= len(fit_routes):
                    break
                worst_perm = fit_routes[-i][2]
                worst_routes, worst_lengths = split_permutation(worst_perm)
                worst_routes, worst_lengths = worst_ruin_recreate(worst_routes, worst_lengths, fraction=ruin_fraction)
                new_perm = []
                for route in worst_routes:
                    new_perm.extend(route[1:-1])
                # replace corresponding worst in new_pop
                worst_fit = float('-inf')
                worst_idx = None
                for idx, p in enumerate(new_pop):
                    fit, _ = evaluate(p)
                    if fit > worst_fit:
                        worst_fit = fit
                        worst_idx = idx
                if worst_idx is not None:
                    new_pop[worst_idx] = new_perm
            stagnation = 0
        else:
            if gen > 0 and fit_routes[0][0] >= best_fitness:
                stagnation += 1
            else:
                stagnation = 0
        population = new_pop
        # Evaluate and update best
        for perm in population:
            fit, routes = evaluate(perm)
            if fit < best_fitness:
                best_fitness = fit
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
    return best_routes