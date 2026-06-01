import numpy as np
import random

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

    def feasible(routes):
        # Check each route starts and ends at 0
        for r in routes:
            if r[0] != 0 or r[-1] != 0:
                return False
        # Check all customers assigned exactly once
        all_cust = []
        for r in routes:
            all_cust.extend(r[1:-1])
        if sorted(all_cust) != customers:
            return False
        return True

    def random_initial_solution():
        # Random assignment ensuring feasibility
        routes = [[0, 0] for _ in range(truck_count)]
        cust_list = customers[:]
        random.shuffle(cust_list)
        # Greedy insertion to min max route length
        for cust in cust_list:
            best_max = float('inf')
            best_t = -1
            best_p = -1
            for t in range(truck_count):
                route = routes[t]
                for p in range(1, len(route)):
                    new_route = route[:p] + [cust] + route[p:]
                    new_len = route_length(new_route)
                    new_max = new_len
                    for tt in range(truck_count):
                        if tt != t:
                            new_max = max(new_max, route_length(routes[tt]))
                    if new_max < best_max or (new_max == best_max and (t < best_t or (t == best_t and p < best_p))):
                        best_max = new_max
                        best_t = t
                        best_p = p
            routes[best_t].insert(best_p, cust)
        lengths = [route_length(r) for r in routes]
        return routes, lengths

    def repair(routes):
        # Ensure feasibility: each route starts and ends at 0; all customers exactly once
        # Already guaranteed by operations, but for safety
        return routes

    def calculate_lengths(routes):
        return [route_length(r) for r in routes]

    best_max = float('inf')
    best_routes = None

    def update_best(routes):
        nonlocal best_max, best_routes
        lengths = [route_length(r) for r in routes]
        m = max(lengths)
        if m < best_max:
            best_max = m
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    pop_size = min(50, n)
    max_gen = 5 * n
    stagnation_limit = int(0.15 * max_gen)
    restart_limit = int(0.3 * max_gen)

    population = []
    for _ in range(pop_size):
        routes, lengths = random_initial_solution()
        population.append((max(lengths), routes))
        update_best(routes)
    population.sort(key=lambda x: x[0])

    no_improve_gen = 0
    for gen in range(max_gen):
        # Adaptive mutation probability
        mutation_prob = 0.2 + 0.15 * (no_improve_gen / stagnation_limit) if no_improve_gen < stagnation_limit else 0.35

        # Tournament selection
        idx1 = random.sample(range(pop_size), 3)
        idx2 = random.sample(range(pop_size), 3)
        parent1 = min([population[i] for i in idx1], key=lambda x: x[0])
        parent2 = min([population[i] for i in idx2], key=lambda x: x[0])
        routes1 = parent1[1]
        routes2 = parent2[1]

        # Route crossover: select a subset of routes from parent1, assign their customers to parent2's routes
        # Implementation: pick a random subset of customers from parent1, then insert them into parent2 routes using min-max heuristic
        crossover_cust = []
        for r in routes1:
            if random.random() < 0.5:
                crossover_cust.extend(r[1:-1])
        if not crossover_cust:
            crossover_cust = [random.choice(customers)]
        # Remove these customers from parent2 routes
        child_routes = [list(r) for r in routes2]
        for r in child_routes:
            for c in crossover_cust:
                if c in r:
                    r.remove(c)
        # Now insert crossover_cust using min-max heuristic
        custs = crossover_cust[:]
        random.shuffle(custs)
        for cust in custs:
            best_max = float('inf')
            best_t = -1
            best_p = -1
            for t in range(truck_count):
                route = child_routes[t]
                for p in range(1, len(route)):
                    new_route = route[:p] + [cust] + route[p:]
                    new_len = route_length(new_route)
                    new_max = new_len
                    for tt in range(truck_count):
                        if tt != t:
                            new_max = max(new_max, route_length(child_routes[tt]))
                    if new_max < best_max or (new_max == best_max and (t < best_t or (t == best_t and p < best_p))):
                        best_max = new_max
                        best_t = t
                        best_p = p
            child_routes[best_t].insert(best_p, cust)
        child_routes = repair(child_routes)

        # Mutation: relocate or swap
        if random.random() < mutation_prob:
            # Relocate a random customer
            t = random.randrange(truck_count)
            if len(child_routes[t]) > 2:
                idx = random.randint(1, len(child_routes[t])-2)
                cust = child_routes[t][idx]
                new_route = child_routes[t][:idx] + child_routes[t][idx+1:]
                child_routes[t] = new_route
                # Insert into best position in another route
                best_max = float('inf')
                best_t = -1
                best_p = -1
                for tt in range(truck_count):
                    if tt == t:
                        continue
                    route = child_routes[tt]
                    for p in range(1, len(route)):
                        new_route2 = route[:p] + [cust] + route[p:]
                        new_len = route_length(new_route2)
                        new_max = new_len
                        for ttt in range(truck_count):
                            if ttt != tt:
                                new_max = max(new_max, route_length(child_routes[ttt]))
                        if new_max < best_max or (new_max == best_max and (tt < best_t or (tt == best_t and p < best_p))):
                            best_max = new_max
                            best_t = tt
                            best_p = p
                child_routes[best_t].insert(best_p, cust)

        # Local search with tabu
        child_lengths = [route_length(r) for r in child_routes]
        tabu_set = set()
        tabu_list = []
        tenure = 5
        max_iter_local = 10 * (n + truck_count)
        for _ in range(max_iter_local):
            # Evaluate best move (relocate or 2-opt)
            best_move = None
            best_new_max = max(child_lengths)
            # Relocate
            for t1 in range(truck_count):
                route1 = child_routes[t1]
                if len(route1) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    cust = route1[i]
                    new_route1 = route1[:i] + route1[i+1:]
                    new_len1 = route_length(new_route1)
                    for t2 in range(truck_count):
                        if t2 == t1:
                            continue
                        route2 = child_routes[t2]
                        for j in range(1, len(route2)):
                            new_route2 = route2[:j] + [cust] + route2[j:]
                            new_len2 = route_length(new_route2)
                            new_max = new_len1
                            for tt in range(truck_count):
                                if tt == t1:
                                    if new_max < new_len1:
                                        new_max = new_len1
                                elif tt == t2:
                                    if new_max < new_len2:
                                        new_max = new_len2
                                else:
                                    if new_max < child_lengths[tt]:
                                        new_max = child_lengths[tt]
                            move = ('relocate', t1, i, t2, j, cust)
                            if move in tabu_set:
                                continue
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_move = move
            # 2-opt intra-route
            for t in range(truck_count):
                route = child_routes[t]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_len = route_length(new_route)
                        new_max = new_len
                        for tt in range(truck_count):
                            if tt != t:
                                if new_max < child_lengths[tt]:
                                    new_max = child_lengths[tt]
                        move = ('2opt', t, i, j)
                        if move in tabu_set:
                            continue
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = move
            if best_move is None or best_new_max >= max(child_lengths):
                break
            # Apply move
            if best_move[0] == 'relocate':
                _, t1, i, t2, j, cust = best_move
                child_routes[t1] = child_routes[t1][:i] + child_routes[t1][i+1:]
                child_routes[t2] = child_routes[t2][:j] + [cust] + child_routes[t2][j:]
            else:
                _, t, i, j = best_move
                child_routes[t] = child_routes[t][:i] + child_routes[t][i:j+1][::-1] + child_routes[t][j+1:]
            child_lengths = [route_length(r) for r in child_routes]
            # Update tabu
            tabu_list.append(best_move)
            tabu_set.add(best_move)
            if len(tabu_list) > tenure:
                old = tabu_list.pop(0)
                tabu_set.discard(old)
            update_best(child_routes)

        child_max = max(child_lengths)
        update_best(child_routes)

        # Replace worst if better
        if child_max < population[-1][0]:
            population[-1] = (child_max, [list(r) for r in child_routes])
            population.sort(key=lambda x: x[0])
            if child_max < best_max:
                no_improve_gen = 0
            else:
                no_improve_gen += 1
        else:
            no_improve_gen += 1

        # Stagnation restart
        if no_improve_gen >= stagnation_limit:
            restart_count = int(0.3 * pop_size)
            for i in range(pop_size - restart_count, pop_size):
                routes, _ = random_initial_solution()
                population[i] = (max(route_length(r) for r in routes), routes)
                update_best(routes)
            population.sort(key=lambda x: x[0])
            no_improve_gen = 0

    return best_routes