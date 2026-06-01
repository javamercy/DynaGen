import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def greedy_construction(perm):
        # Build routes from permutation using min-max insertion
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        for cust in perm:
            best_max = float('inf')
            best_truck = -1
            best_pos = -1
            for t in range(truck_count):
                for pos in range(1, len(routes[t])):
                    new_route = routes[t][:pos] + [cust] + routes[t][pos:]
                    new_dist = route_distance(new_route)
                    new_max = new_dist
                    for k in range(truck_count):
                        if k != t:
                            new_max = max(new_max, route_dists[k])
                    if new_max < best_max:
                        best_max = new_max
                        best_truck = t
                        best_pos = pos
            # deterministic tie: already handled by first encountered
            routes[best_truck] = routes[best_truck][:best_pos] + [cust] + routes[best_truck][best_pos:]
            route_dists[best_truck] = route_distance(routes[best_truck])
        return routes, route_dists

    def decode(perm):
        routes, _ = greedy_construction(perm)
        max_len = max(route_distance(r) for r in routes)
        return routes, max_len

    # Tabu search improvement (short version)
    def tabu_improve(routes):
        best_routes = [list(r) for r in routes]
        best_max = max(route_distance(r) for r in routes)
        current_routes = [list(r) for r in routes]
        current_dists = [route_distance(r) for r in current_routes]
        tabu_tenure = 5
        tabu_list = []
        tabu_set = set()
        max_iter = max(20, (n-1) // 2)
        for _ in range(max_iter):
            best_move = None
            best_new_max = float('inf')
            best_tie = None
            # Relocate
            for t1 in range(truck_count):
                route1 = current_routes[t1]
                if len(route1) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    cust = route1[i]
                    new_route1 = route1[:i] + route1[i+1:]
                    dist1_new = route_distance(new_route1)
                    for t2 in range(truck_count):
                        if t1 == t2:
                            continue
                        route2 = current_routes[t2]
                        for j in range(1, len(route2)):
                            new_route2 = route2[:j] + [cust] + route2[j:]
                            dist2_new = route_distance(new_route2)
                            new_max = max(dist1_new, dist2_new)
                            for k in range(truck_count):
                                if k != t1 and k != t2:
                                    new_max = max(new_max, current_dists[k])
                            is_tabu = (cust, t2, t1) in tabu_set
                            if is_tabu and new_max >= best_max:
                                continue
                            tie = (new_max, 0, t1, i, t2, j)
                            if best_tie is None or tie < best_tie:
                                best_new_max = new_max
                                best_move = ('relocate', t1, i, t2, j, cust)
                                best_tie = tie
            # Swap
            for t1 in range(truck_count):
                route1 = current_routes[t1]
                if len(route1) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    cust1 = route1[i]
                    for t2 in range(t1+1, truck_count):
                        route2 = current_routes[t2]
                        if len(route2) <= 2:
                            continue
                        for j in range(1, len(route2)-1):
                            cust2 = route2[j]
                            new_route1 = route1[:i] + [cust2] + route1[i+1:]
                            new_route2 = route2[:j] + [cust1] + route2[j+1:]
                            dist1_new = route_distance(new_route1)
                            dist2_new = route_distance(new_route2)
                            new_max = max(dist1_new, dist2_new)
                            for k in range(truck_count):
                                if k != t1 and k != t2:
                                    new_max = max(new_max, current_dists[k])
                            is_tabu = ((cust1, cust2, t1, t2) in tabu_set) or ((cust2, cust1, t2, t1) in tabu_set)
                            if is_tabu and new_max >= best_max:
                                continue
                            tie = (new_max, 1, t1, i, t2, j)
                            if best_tie is None or tie < best_tie:
                                best_new_max = new_max
                                best_move = ('swap', t1, i, t2, j, cust1, cust2)
                                best_tie = tie
            # 2-opt
            for t in range(truck_count):
                route = current_routes[t]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        new_max = new_dist
                        for k in range(truck_count):
                            if k != t:
                                new_max = max(new_max, current_dists[k])
                        tie = (new_max, 2, t, i, j)
                        if best_tie is None or tie < best_tie:
                            best_new_max = new_max
                            best_move = ('2opt', t, i, j, new_route)
                            best_tie = tie
            if best_move is None:
                break
            # Apply move
            if best_move[0] == 'relocate':
                _, t1, i, t2, j, cust = best_move
                current_routes[t1] = current_routes[t1][:i] + current_routes[t1][i+1:]
                current_routes[t2] = current_routes[t2][:j] + [cust] + current_routes[t2][j:]
            elif best_move[0] == 'swap':
                _, t1, i, t2, j, cust1, cust2 = best_move
                current_routes[t1][i] = cust2
                current_routes[t2][j] = cust1
            else: # 2opt
                _, t, i, j, new_route = best_move
                current_routes[t] = new_route
            # Update distances
            for t in range(truck_count):
                current_dists[t] = route_distance(current_routes[t])
            new_max = max(current_dists)
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in current_routes]
                report_best_vrp(best_routes)
            # Manage tabu
            if best_move[0] == 'relocate':
                tabu_entry = (best_move[5], best_move[3], best_move[1])
                tabu_list.append(tabu_entry)
                tabu_set.add(tabu_entry)
            elif best_move[0] == 'swap':
                tabu_entry1 = (best_move[5], best_move[6], best_move[1], best_move[3])
                tabu_entry2 = (best_move[6], best_move[5], best_move[3], best_move[1])
                tabu_list.append(tabu_entry1)
                tabu_set.add(tabu_entry1)
                tabu_list.append(tabu_entry2)
                tabu_set.add(tabu_entry2)
            if len(tabu_list) > tabu_tenure:
                old = tabu_list.pop(0)
                tabu_set.discard(old)
        return best_routes, best_max

    # Initial solution via greedy (using random order as seed)
    def random_perm():
        perm = customers[:]
        random.shuffle(perm)
        return perm

    # Best solution tracking
    best_max = float('inf')
    best_routes = None
    def report_best_vrp(routes):
        nonlocal best_max, best_routes
        m = max(route_distance(r) for r in routes)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]

    # Population
    pop_size = min(30, n)
    max_gen = 3 * n
    mutation_prob = 0.15

    population = []
    for _ in range(pop_size):
        perm = random_perm()
        routes, max_len = decode(perm)
        report_best_vrp(routes)
        # Improve initial solutions with tabu
        routes_imp, max_imp = tabu_improve(routes)
        report_best_vrp(routes_imp)
        # Store as (max_len, perm) but note that improved max may be better
        # We'll store the improved max and the original perm (or improved? keep perm for recombination)
        # Actually we need the permutation to do crossover; but after tabu we have routes, not a permutation.
        # To keep GA simple, we decode permutation for each individual; tabu is only applied to offspring after crossover.
        # So we store permutation only.
        population.append((max_len, perm))
    population.sort(key=lambda x: x[0])

    for _ in range(max_gen):
        # Binary tournament selection
        idx1 = random.sample(range(pop_size), 2)
        idx2 = random.sample(range(pop_size), 2)
        parent1 = population[idx1[0]] if population[idx1[0]] <= population[idx1[1]] else population[idx1[1]]
        parent2 = population[idx2[0]] if population[idx2[0]] <= population[idx2[1]] else population[idx2[1]]
        p1 = parent1[1]
        p2 = parent2[1]

        # PMX crossover
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
                gene = p2[i]
                while gene in mapping:
                    gene = mapping[gene]
                child[i] = gene
        used = set(child)
        remaining = [c for c in customers if c not in used]
        for i in range(n_cust):
            if child[i] is None:
                child[i] = remaining.pop()

        # Inversion mutation
        if random.random() < mutation_prob:
            i = random.randint(0, n_cust-1)
            j = random.randint(0, n_cust-1)
            if i > j:
                i, j = j, i
            child[i:j+1] = reversed(child[i:j+1])

        # Decode and apply tabu improvement
        routes_child, max_child = decode(child)
        routes_imp, max_imp = tabu_improve(routes_child)
        report_best_vrp(routes_imp)
        # Use improved max for comparison
        if max_imp < population[-1][0]:
            # Replace worst
            population[-1] = (max_imp, child)
            population.sort(key=lambda x: x[0])

    return best_routes