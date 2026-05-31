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

    def evaluate(routes):
        dists = [route_distance(r) for r in routes]
        return dists, max(dists), sum(dists)

    def greedy_insertion(routes, unassigned):
        dists = [route_distance(r) for r in routes]
        current_max = max(dists)
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
                    new_max = max(current_max, new_dist)
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
            if dists[best_truck] > current_max:
                current_max = dists[best_truck]
        return routes, dists, max(dists), sum(dists)

    # Farthest-first initial construction (deterministic)
    customers = sorted(range(1, n), key=lambda c: -dist[0][c])
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for cust in customers:
        best_truck = None
        best_pos = None
        best_max = float('inf')
        best_total = float('inf')
        for t, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_dist = route_dists[t] + insertion_delta(route, pos, cust)
                new_max = max(route_dists[:t] + [new_dist] + route_dists[t+1:])
                new_total = sum(route_dists) + insertion_delta(route, pos, cust)
                if new_max < best_max or (new_max == best_max and new_total < best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        route = routes[best_truck]
        routes[best_truck] = route[:best_pos] + [cust] + route[best_pos:]
        route_dists[best_truck] += insertion_delta(route, best_pos, cust)
    init_routes = [list(r) for r in routes]
    init_max = max(route_dists)
    init_total = sum(route_dists)

    # Generate initial population
    pop_size = 30
    population = []
    # First individual: farthest-first
    dists, max_dist, total_dist = evaluate(init_routes)
    population.append((init_routes, dists, max_dist, total_dist))
    best_routes = [list(r) for r in init_routes]
    best_dists = list(dists)
    best_max = max_dist
    best_total = total_dist
    report_best_vrp(best_routes)
    # Rest: randomized greedy insertion
    for _ in range(pop_size - 1):
        cust_order = list(range(1, n))
        random.shuffle(cust_order)
        routes = [[0, 0] for _ in range(truck_count)]
        routes, dists, max_d, total_d = greedy_insertion(routes, cust_order)
        # Ensure it's not duplicate (by max and total)
        dup = False
        for (r, d, m, t) in population:
            if abs(m - max_d) < 1e-9 and abs(t - total_d) < 1e-9:
                dup = True
                break
        if not dup:
            population.append((routes, dists, max_d, total_d))
        if max_d < best_max - 1e-9 or (abs(max_d - best_max) < 1e-9 and total_d < best_total):
            best_max = max_d
            best_total = total_d
            best_routes = [list(r) for r in routes]
            best_dists = list(dists)
            report_best_vrp(best_routes)
    # Fill population if duplicates reduced size
    while len(population) < pop_size:
        cust_order = list(range(1, n))
        random.shuffle(cust_order)
        routes = [[0, 0] for _ in range(truck_count)]
        routes, dists, max_d, total_d = greedy_insertion(routes, cust_order)
        dup = False
        for (r, d, m, t) in population:
            if abs(m - max_d) < 1e-9 and abs(t - total_d) < 1e-9:
                dup = True
                break
        if not dup:
            population.append((routes, dists, max_d, total_d))

    # GA parameters
    gen_max = min(200, 20 * n)
    tournament_size = 3
    crossover_rate = 0.8
    mutation_rate = 0.3

    # Helper: tournament selection
    def tournament_select():
        idxs = random.sample(range(len(population)), tournament_size)
        best_idx = idxs[0]
        for i in idxs[1:]:
            _, _, m1, t1 = population[best_idx]
            _, _, m2, t2 = population[i]
            if m2 < m1 - 1e-9 or (abs(m2 - m1) < 1e-9 and t2 < t1):
                best_idx = i
        return population[best_idx]

    def crossover(p1, p2):
        # p1, p2: (routes, dists, max, total)
        routes1 = p1[0]
        routes2 = p2[0]
        # Step 1: copy random number of routes from p1
        k = random.randint(1, truck_count - 1)
        indices = random.sample(range(truck_count), k)
        child = [[0, 0] for _ in range(truck_count)]
        assigned = set()
        for idx in indices:
            route = list(routes1[idx])
            child[idx] = route
            for c in route[1:-1]:
                assigned.add(c)
        # Step 2: copy non-conflicting routes from p2 into empty slots
        p2_indices = list(range(truck_count))
        random.shuffle(p2_indices)
        for idx in p2_indices:
            route = routes2[idx]
            custs = route[1:-1]
            if all(c not in assigned for c in custs):
                empty = [i for i, r in enumerate(child) if len(r) == 2 and r[0] == 0 and r[1] == 0]
                if empty:
                    child[empty[0]] = list(route)
                    for c in custs:
                        assigned.add(c)
        # Step 3: greedily insert unassigned
        unassigned = [c for c in range(1, n) if c not in assigned]
        random.shuffle(unassigned)
        child, dists, max_d, total_d = greedy_insertion(child, unassigned)
        return child, dists, max_d, total_d

    def mutate(offspring):
        routes = offspring[0]
        dists = offspring[1]
        # Random relocation: pick a customer, remove, reinsert randomly
        cust = random.randint(1, n-1)
        # find its current position
        found = False
        for t, route in enumerate(routes):
            for pos, c in enumerate(route):
                if c == cust:
                    # remove
                    new_route = route[:pos] + route[pos+1:]
                    routes[t] = new_route
                    # update dists
                    old_dist = dists[t]
                    # new dist is route_distance of new_route
                    new_d = route_distance(new_route)
                    dists[t] = new_d
                    found = True
                    break
            if found:
                break
        if not found:
            return offspring  # should not happen
        # Insert into a random route at random position
        t = random.randint(0, truck_count-1)
        route = routes[t]
        pos = random.randint(1, len(route)-1)
        new_route = route[:pos] + [cust] + route[pos:]
        routes[t] = new_route
        dists[t] = route_distance(new_route)
        max_d = max(dists)
        total_d = sum(dists)
        return routes, dists, max_d, total_d

    for gen in range(gen_max):
        # Select parents
        p1 = tournament_select()
        p2 = tournament_select()
        if random.random() < crossover_rate:
            offspring = crossover(p1, p2)
        else:
            # copy one parent
            p = p1 if random.random() < 0.5 else p2
            offspring = ( [list(r) for r in p[0]], list(p[1]), p[2], p[3] )
        if random.random() < mutation_rate:
            offspring = mutate(offspring)
        # Evaluate (already done in crossover/mutate)
        child_routes, child_dists, child_max, child_total = offspring
        # Replace worst if better
        # Find worst index
        worst_idx = 0
        worst_max = -1
        worst_total = -1
        for i, (r, d, m, t) in enumerate(population):
            if m > worst_max + 1e-9 or (abs(m - worst_max) < 1e-9 and t > worst_total):
                worst_max = m
                worst_total = t
                worst_idx = i
        if child_max < worst_max - 1e-9 or (abs(child_max - worst_max) < 1e-9 and child_total < worst_total):
            population[worst_idx] = offspring
        # Update best
        if child_max < best_max - 1e-9 or (abs(child_max - best_max) < 1e-9 and child_total < best_total):
            best_max = child_max
            best_total = child_total
            best_routes = [list(r) for r in child_routes]
            best_dists = list(child_dists)
            report_best_vrp(best_routes)

    # Post-optimization: 2-opt on best solution
    max_opt_iter = 200
    for _ in range(max_opt_iter):
        improved = False
        for t, route in enumerate(best_routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_dists[t] - 1e-9:
                        new_max = max(best_dists[:t] + [new_dist] + best_dists[t+1:])
                        new_total = sum(best_dists[:t]) + new_dist + sum(best_dists[t+1:])
                        if new_max < best_max - 1e-9 or (abs(new_max - best_max) < 1e-9 and new_total < best_total):
                            best_routes[t] = new_route
                            best_dists[t] = new_dist
                            best_max = new_max
                            best_total = new_total
                            report_best_vrp(best_routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_routes