import numpy as np
import random
from math import exp

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        total = 0.0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def max_route_len(routes):
        return max(route_length(r) for r in routes) if routes else float('inf')

    def construct_solution(shuffle_order=True):
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        if shuffle_order:
            random.shuffle(unassigned)
        while unassigned:
            candidates = []
            for cust in unassigned:
                insert_info = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_info.append((new_max, cost, r_idx, pos))
                insert_info.sort(key=lambda x: (x[0], x[1]))
                best = insert_info[0]
                second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                regret = second[0] - best[0]
                candidates.append((best[0], regret, best[1], best[2], best[3], cust))
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2], x[5]))
            chosen = candidates[0]
            _, _, _, r_idx, pos, cust = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        return routes

    def ruin_recreate(routes, perturb_size=0.1):
        route_lens = [(route_length(r), idx) for idx, r in enumerate(routes)]
        route_lens.sort(reverse=True)
        num_to_remove = max(1, int((n-1) * perturb_size))
        removed = []
        for _, r_idx in route_lens:
            route = routes[r_idx]
            if len(route) <= 2:
                continue
            can_remove = min(num_to_remove - len(removed), len(route)-2)
            if can_remove <= 0:
                break
            remove_set = set(random.sample(route[1:-1], can_remove))
            for cust in remove_set:
                removed.append((r_idx, cust))
            routes[r_idx] = [x for x in route if x not in remove_set]
            if len(removed) >= num_to_remove:
                break
        unassigned = [cust for _, cust in removed]
        random.shuffle(unassigned)
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_data = None
            for cust in unassigned:
                insert_info = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        new_len = route_length(route) + cost
                        other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                        new_max = max(new_len, *other_lens)
                        insert_info.append((new_max, cost, r_idx, pos))
                if not insert_info:
                    continue
                insert_info.sort(key=lambda x: (x[0], x[1]))
                best = insert_info[0]
                second = insert_info[1] if len(insert_info) > 1 else (best[0] + 1e9, best[1] + 1e9, -1, -1)
                regret = second[0] - best[0]
                if best_cust is None or regret > best_regret or (regret == best_regret and best[1] < best_data[1]):
                    best_cust = cust
                    best_regret = regret
                    best_data = (best[0], best[1], best[2], best[3])
            if best_cust is None:
                break
            _, _, r_idx, pos = best_data
            routes[r_idx].insert(pos, best_cust)
            unassigned.remove(best_cust)
        return routes

    # Population size and generations bounded by instance size
    pop_size = min(20, max(10, n // 2))
    max_generations = n * truck_count // pop_size
    if max_generations < 10:
        max_generations = 10

    # Initialize population
    population = []
    for _ in range(pop_size):
        routes = construct_solution(shuffle_order=True)
        fitness = max_route_len(routes)
        population.append((fitness, routes))
    population.sort(key=lambda x: x[0])
    best_fitness = population[0][0]
    best_routes = [r[:] for r in population[0][1]]
    report_best_vrp(best_routes)

    for gen in range(max_generations):
        new_population = []
        # Elitism: keep best 2
        new_population.extend(population[:2])
        while len(new_population) < pop_size:
            # Tournament selection
            i1, i2 = random.sample(range(pop_size), 2)
            parent1 = population[i1][1]
            parent2 = population[i2][1]
            # Crossover probability 0.8
            if random.random() < 0.8:
                offspring = crossover(parent1, parent2)
            else:
                offspring = [r[:] for r in parent1]
            # Mutation probability 0.1
            if random.random() < 0.1:
                offspring = ruin_recreate(offspring, perturb_size=0.1)
            # Ensure all customers present
            # Repair if necessary: find missing/duplicate
            all_custs = set(range(1, n))
            assigned = set()
            for route in offspring:
                for cust in route:
                    if cust != 0:
                        assigned.add(cust)
            missing = all_custs - assigned
            if missing:
                # Simple repair: insert missing using regret
                for cust in missing:
                    best_insert = None
                    best_cost = float('inf')
                    for r_idx, route in enumerate(offspring):
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                            if cost < best_cost:
                                best_cost = cost
                                best_insert = (r_idx, pos)
                    if best_insert:
                        r_idx, pos = best_insert
                        offspring[r_idx].insert(pos, cust)
            # Also check duplicates (should not happen, but safe)
            seen = set()
            for r_idx, route in enumerate(offspring):
                new_route = [0]
                for cust in route[1:-1]:
                    if cust not in seen:
                        seen.add(cust)
                        new_route.append(cust)
                    # else skip
                new_route.append(0)
                offspring[r_idx] = new_route
            # Ensure all missing added? Already handled.
            # Evaluate fitness
            fitness = max_route_len(offspring)
            new_population.append((fitness, offspring))
            # Update best
            if fitness < best_fitness:
                best_fitness = fitness
                best_routes = [r[:] for r in offspring]
                report_best_vrp(best_routes)
        # Sort and keep best pop_size
        new_population.sort(key=lambda x: x[0])
        population = new_population[:pop_size]
        # Update best from population
        if population[0][0] < best_fitness:
            best_fitness = population[0][0]
            best_routes = [r[:] for r in population[0][1]]
            report_best_vrp(best_routes)

    # Detailed crossover implementation
    def crossover(parent1, parent2):
        # Route-based exchange: randomly select a subset of routes from parent1
        num_routes = len(parent1)
        selected_indices = set(random.sample(range(num_routes), random.randint(1, max(1, num_routes//2))))
        # Build partial offspring with selected routes
        offspring = [None] * num_routes
        assigned_customers = set()
        for idx in selected_indices:
            route = parent1[idx][:]
            offspring[idx] = route
            for cust in route:
                if cust != 0:
                    assigned_customers.add(cust)
        # Now fill remaining routes from parent2, but skip customers already assigned
        # We'll copy routes from parent2 that don't conflict, then distribute remaining
        remaining_routes = []
        for idx in range(num_routes):
            if idx not in selected_indices:
                route = parent2[idx][:]
                # Remove already assigned customers
                new_route = [0]
                for cust in route[1:-1]:
                    if cust not in assigned_customers:
                        new_route.append(cust)
                new_route.append(0)
                offspring[idx] = new_route
        # Now some customers might be missing from both parents? Actually assigned_customers are from parent1,
        # we need to collect all customers from parent2 that are not assigned yet.
        all_custs = set(range(1, n))
        assigned_now = set()
        for route in offspring:
            if route:
                for cust in route:
                    if cust != 0:
                        assigned_now.add(cust)
        missing = all_custs - assigned_now
        # Insert missing using regret heuristic
        missing_list = list(missing)
        random.shuffle(missing_list)
        for cust in missing_list:
            best_insert = None
            best_cost = float('inf')
            for r_idx, route in enumerate(offspring):
                if route is None:
                    continue
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if cost < best_cost:
                        best_cost = cost
                        best_insert = (r_idx, pos)
            if best_insert:
                r_idx, pos = best_insert
                offspring[r_idx].insert(pos, cust)
        # Handle empty routes: ensure [0,0]
        for idx in range(num_routes):
            if offspring[idx] is None or len(offspring[idx]) <= 2:
                offspring[idx] = [0, 0]
        return offspring

    return best_routes