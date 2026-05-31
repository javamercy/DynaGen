import numpy as np
import math
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)
    dist = distance_matrix

    def route_distance(route):
        if len(route) <= 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    def insertion_delta(route, pos, cust):
        prev = route[pos-1]
        nxt = route[pos]
        return dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt]

    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        dists = [0.0] * truck_count
        for cust in perm:
            best_truck = None
            best_pos = None
            best_max = float('inf')
            best_total = float('inf')
            for t, route in enumerate(routes):
                for pos in range(1, len(route)):
                    delta = insertion_delta(route, pos, cust)
                    new_dist = dists[t] + delta
                    new_max = max(dists[:t] + [new_dist] + dists[t+1:])
                    new_total = sum(dists) + delta
                    if new_max < best_max or (new_max == best_max and new_total < best_total):
                        best_max = new_max
                        best_total = new_total
                        best_truck = t
                        best_pos = pos
            route = routes[best_truck]
            routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
            dists[best_truck] += insertion_delta(route, best_pos, cust)
        return routes, dists

    def order_crossover(p1, p2):
        size = len(p1)
        a = random.randrange(size)
        b = random.randrange(size)
        if a > b:
            a, b = b, a
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        current = 0
        for i in range(size):
            if child[i] is None:
                while p2[current] in child:
                    current += 1
                child[i] = p2[current]
                current += 1
        return child

    def worst_removal(routes, dists, num_removals):
        all_contribs = []
        for t, route in enumerate(routes):
            if len(route) <= 2:
                continue
            base = dists[t]
            for i in range(1, len(route)-1):
                prev = route[i-1]
                nxt = route[i+1]
                with_ = dist[prev, route[i]] + dist[route[i], nxt]
                without = dist[prev, nxt]
                contrib = with_ - without
                all_contribs.append((contrib, t, i, route[i]))
        all_contribs.sort(key=lambda x: -x[0])
        to_remove = set()
        for _, t, i, cust in all_contribs[:num_removals]:
            to_remove.add(cust)
        new_routes = []
        new_dists = []
        for t, route in enumerate(routes):
            new_route = [0] + [c for c in route[1:-1] if c not in to_remove] + [0]
            new_routes.append(new_route)
            new_dists.append(route_distance(new_route))
        return list(to_remove), new_routes, new_dists

    def greedy_repair(routes, dists, unassigned):
        routes = [list(r) for r in routes]
        dists = list(dists)
        unassigned = list(unassigned)
        current_max_local = max(dists)
        for cust in unassigned:
            best_truck = None
            best_pos = None
            best_new_max = float('inf')
            best_new_total = float('inf')
            best_delta = None
            for t, route in enumerate(routes):
                old_dist = dists[t]
                for pos in range(1, len(route)):
                    delta = insertion_delta(route, pos, cust)
                    new_dist = old_dist + delta
                    new_max = max(current_max_local, new_dist)
                    new_total = sum(dists) + delta
                    if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_truck = t
                        best_pos = pos
                        best_delta = delta
            route = routes[best_truck]
            routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
            dists[best_truck] += best_delta
            if dists[best_truck] > current_max_local:
                current_max_local = dists[best_truck]
        return routes, dists

    def regret2_repair(routes, dists, unassigned):
        routes = [list(r) for r in routes]
        dists = list(dists)
        unassigned = list(unassigned)
        current_max_local = max(dists)
        while unassigned:
            best_info = None
            for cust in unassigned:
                best_max_val = float('inf')
                best_total_val = float('inf')
                best_truck = None
                best_pos = None
                best_delta = None
                second_best_max = float('inf')
                second_best_total = float('inf')
                for t, route in enumerate(routes):
                    old_dist = dists[t]
                    for pos in range(1, len(route)):
                        delta = insertion_delta(route, pos, cust)
                        new_dist = old_dist + delta
                        new_max = max(current_max_local, new_dist)
                        new_total = sum(dists) + delta
                        if new_max < best_max_val or (new_max == best_max_val and new_total < best_total_val):
                            second_best_max = best_max_val
                            second_best_total = best_total_val
                            best_max_val = new_max
                            best_total_val = new_total
                            best_truck = t
                            best_pos = pos
                            best_delta = delta
                        elif new_max < second_best_max or (new_max == second_best_max and new_total < second_best_total):
                            second_best_max = new_max
                            second_best_total = new_total
                if second_best_max == float('inf'):
                    regret = float('inf')
                else:
                    regret = second_best_max - best_max_val
                if best_info is None or regret > best_info[0] or (regret == best_info[0] and (best_max_val < best_info[1] or (best_max_val == best_info[1] and cust < best_info[4]))):
                    best_info = (regret, best_max_val, best_total_val, cust, best_truck, best_pos, best_delta)
            regret, best_max_val, best_total_val, cust, best_truck, best_pos, best_delta = best_info
            route = routes[best_truck]
            routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
            dists[best_truck] += best_delta
            if dists[best_truck] > current_max_local:
                current_max_local = dists[best_truck]
            unassigned.remove(cust)
        return routes, dists

    # Initialize population
    pop_size = min(20, max(10, 2 * truck_count))
    population = []
    for _ in range(pop_size):
        perm = list(range(1, n))
        random.shuffle(perm)
        routes, dists = decode(perm)
        max_dist = max(dists)
        total_dist = sum(dists)
        population.append((perm, routes, dists, max_dist, total_dist))
    best_idx = min(range(pop_size), key=lambda i: (population[i][3], population[i][4]))
    best_perm, best_routes, best_dists, best_max, best_total = population[best_idx]
    report_best_vrp(best_routes)

    # GA parameters
    max_gen = min(2000, 10 * n)
    num_elite = 1
    mutation_rate = 0.2
    removal_fraction = 0.25
    num_removals = max(1, int(removal_fraction * (n - 1)))

    for gen in range(max_gen):
        new_pop = []
        # Elitism
        new_pop.append((best_perm, best_routes, best_dists, best_max, best_total))
        while len(new_pop) < pop_size:
            # Tournament selection (size 2)
            i1 = random.randrange(pop_size)
            i2 = random.randrange(pop_size)
            while i2 == i1:
                i2 = random.randrange(pop_size)
            parent1 = population[i1] if (population[i1][3], population[i1][4]) < (population[i2][3], population[i2][4]) else population[i2]
            i1 = random.randrange(pop_size)
            i2 = random.randrange(pop_size)
            while i2 == i1:
                i2 = random.randrange(pop_size)
            parent2 = population[i1] if (population[i1][3], population[i1][4]) < (population[i2][3], population[i2][4]) else population[i2]
            # Crossover
            if random.random() < 0.8:
                child_perm = order_crossover(parent1[0], parent2[0])
            else:
                child_perm = parent1[0][:]
            # Mutation: ruin and recreate
            if random.random() < mutation_rate:
                routes, dists = decode(child_perm)
                to_remove, partial_routes, partial_dists = worst_removal(routes, dists, num_removals)
                repair_op = random.randint(0, 1)
                if repair_op == 0:
                    new_routes, new_dists = greedy_repair(partial_routes, partial_dists, to_remove)
                else:
                    new_routes, new_dists = regret2_repair(partial_routes, partial_dists, to_remove)
                # Reconstruct permutation from routes
                new_perm = []
                for r in new_routes:
                    for c in r[1:-1]:
                        new_perm.append(c)
                child_perm = new_perm
            # Decode child
            routes, dists = decode(child_perm)
            max_dist = max(dists)
            total_dist = sum(dists)
            new_pop.append((child_perm, routes, dists, max_dist, total_dist))
        population = new_pop
        # Update best
        best_idx = min(range(pop_size), key=lambda i: (population[i][3], population[i][4]))
        cand_perm, cand_routes, cand_dists, cand_max, cand_total = population[best_idx]
        if cand_max < best_max - 1e-9 or (abs(cand_max - best_max) < 1e-9 and cand_total < best_total):
            best_max = cand_max
            best_total = cand_total
            best_routes = [list(r) for r in cand_routes]
            best_dists = list(cand_dists)
            report_best_vrp(best_routes)

    return best_routes