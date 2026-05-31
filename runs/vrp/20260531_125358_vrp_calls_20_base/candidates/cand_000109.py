import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))

    def route_distance(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def split_permutation(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        lengths = [0] * truck_count
        for cust in perm:
            best_truck = -1
            best_new_max = float('inf')
            best_pos = -1
            for t in range(truck_count):
                best_inc = float('inf')
                best_p = -1
                for p in range(1, len(routes[t])):
                    inc = distance_matrix[routes[t][p-1], cust] + distance_matrix[cust, routes[t][p]] - distance_matrix[routes[t][p-1], routes[t][p]]
                    if inc < best_inc:
                        best_inc = inc
                        best_p = p
                new_len = lengths[t] + best_inc
                new_max = max(new_len, max(lengths[:t] + lengths[t+1:] + [0]))
                if new_max < best_new_max or (new_max == best_new_max and t < best_truck):
                    best_new_max = new_max
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
                break
        return routes, lengths

    def regret_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        unvisited = set(customers)
        while unvisited:
            best_cust = None
            best_regret = -float('inf')
            best_inc = float('inf')
            best_route = -1
            best_pos = -1
            for cust in unvisited:
                incs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        inc = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]]
                        incs.append((inc, pos, r_idx))
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
                    best_route = r_idx
                    best_pos = pos
            routes[best_route].insert(best_pos, best_cust)
            unvisited.remove(best_cust)
        return routes

    # Adaptive parameter schedules
    pop_size = min(50, n * 2)
    generations = min(100, n * 5)
    crossover_start = 0.9
    crossover_end = 0.5
    mutation_start = 0.2
    mutation_end = 0.01
    elite_count = 2

    # Initial population
    population = []  # list of (perm, fitness, routes)
    for _ in range(pop_size - 1):
        perm = list(customers)
        random.shuffle(perm)
        population.append(perm)
    greedy_routes = regret_construction()
    greedy_perm = []
    for route in greedy_routes:
        greedy_perm.extend(route[1:-1])
    population.append(greedy_perm)

    def evaluate(perm):
        routes, lengths = split_permutation(perm)
        routes, lengths = local_search(routes, lengths)
        return max(lengths), routes

    best_fitness = float('inf')
    best_routes = None
    for perm in population:
        fit, routes = evaluate(perm)
        if fit < best_fitness:
            best_fitness = fit
            best_routes = routes
            report_best_vrp(best_routes)

    # GA loop
    for gen in range(generations):
        # Update adaptive rates
        frac = gen / (generations - 1) if generations > 1 else 0
        crossover_rate = crossover_start + (crossover_end - crossover_start) * frac
        mutation_rate = mutation_start + (mutation_end - mutation_start) * frac

        # Evaluate all
        fitness_list = []
        for perm in population:
            fit, routes = evaluate(perm)
            fitness_list.append((fit, routes, perm))
        fitness_list.sort(key=lambda x: x[0])

        # Elitism
        new_pop = [p[2] for p in fitness_list[:elite_count]]

        # Selection and reproduction
        while len(new_pop) < pop_size:
            # tournament
            idx1 = random.randint(0, len(population)-1)
            idx2 = random.randint(0, len(population)-1)
            idx3 = random.randint(0, len(population)-1)
            best_idx = min([idx1, idx2, idx3], key=lambda i: fitness_list[i][0])
            parent1 = population[best_idx]
            idx1 = random.randint(0, len(population)-1)
            idx2 = random.randint(0, len(population)-1)
            idx3 = random.randint(0, len(population)-1)
            best_idx = min([idx1, idx2, idx3], key=lambda i: fitness_list[i][0])
            parent2 = population[best_idx]

            if random.random() < crossover_rate:
                # order crossover
                perm_len = len(parent1)
                start = random.randint(0, perm_len-1)
                end = random.randint(start+1, perm_len)
                child1 = [None] * perm_len
                child2 = [None] * perm_len
                child1[start:end] = parent1[start:end]
                ch1_pos = end
                for gene in parent2:
                    if gene not in child1:
                        child1[ch1_pos % perm_len] = gene
                        ch1_pos += 1
                child2[start:end] = parent2[start:end]
                ch2_pos = end
                for gene in parent1:
                    if gene not in child2:
                        child2[ch2_pos % perm_len] = gene
                        ch2_pos += 1
            else:
                child1, child2 = parent1[:], parent2[:]

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

        population = new_pop

        # Evaluate new population
        for perm in population:
            fit, routes = evaluate(perm)
            if fit < best_fitness:
                best_fitness = fit
                best_routes = routes
                report_best_vrp(best_routes)

    return best_routes