import numpy as np
import random

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
        return max(route_length(r) for r in routes)

    # --- Initial construction: regret-based with random tie-breaking (from parent) ---
    def construct_solution():
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        while unassigned:
            candidates = []
            for cust in unassigned:
                costs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                        costs.append((cost, r_idx, pos))
                costs.sort(key=lambda x: x[0])
                best_cost = costs[0][0]
                second_cost = costs[1][0] if len(costs) > 1 else best_cost + 1e9
                regret = second_cost - best_cost
                candidates.append((cust, regret, best_cost, costs[0][1], costs[0][2]))
            candidates.sort(key=lambda x: (-x[1], x[2]))  # higher regret first, then lower cost
            k = min(3, len(candidates))
            chosen = random.choice(candidates[:k])
            cust, _, _, r_idx, pos = chosen
            routes[r_idx].insert(pos, cust)
            unassigned.remove(cust)
        return routes

    # --- Crossover: route-based crossover ---
    def crossover(parent1, parent2):
        child = [[0, 0] for _ in range(truck_count)]
        # Randomly select a subset of routes to copy from parent1
        copy_indices = list(range(truck_count))
        random.shuffle(copy_indices)
        num_copy = random.randint(1, truck_count-1) if truck_count > 1 else 1
        selected_routes = copy_indices[:num_copy]
        used_customers = set()
        for idx in selected_routes:
            route = parent1[idx]
            if route[0] == 0 and route[-1] == 0 and len(route) >= 2:
                child[idx] = route[:]
                for cust in route[1:-1]:
                    used_customers.add(cust)
        # Insert remaining customers using cheapest insertion
        unassigned = [c for c in range(1, n) if c not in used_customers]
        for cust in unassigned:
            best_cost = float('inf')
            best_r_idx = 0
            best_pos = 1
            for r_idx in range(truck_count):
                route = child[r_idx]
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if cost < best_cost - 1e-12:
                        best_cost = cost
                        best_r_idx = r_idx
                        best_pos = pos
            child[best_r_idx].insert(best_pos, cust)
        # Fix any route that might be empty (should not happen because we copy routes, but ensure)
        for idx in range(truck_count):
            if len(child[idx]) <= 2 and child[idx][0] != 0:
                child[idx] = [0, 0]
        return child

    # --- Mutation: remove and reinsert some customers ---
    def mutate(routes):
        # Remove up to 10% customers and reinsert greedily
        cust_list = list(range(1, n))
        random.shuffle(cust_list)
        num_perturb = max(1, n // 10)
        to_move = cust_list[:num_perturb]
        for cust in to_move:
            # remove
            for r_idx, route in enumerate(routes):
                if cust in route:
                    routes[r_idx] = [c for c in route if c != cust]
                    break
        # reinsert
        for cust in to_move:
            best_cost = float('inf')
            best_r_idx = 0
            best_pos = 1
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    if cost < best_cost - 1e-12:
                        best_cost = cost
                        best_r_idx = r_idx
                        best_pos = pos
            routes[best_r_idx].insert(best_pos, cust)
        return routes

    # --- Genetic Algorithm ---
    pop_size = min(50, max(10, n // 5))
    population = []
    best_solution = None
    best_fitness = float('inf')

    # Initialize population
    for _ in range(pop_size):
        sol = construct_solution()
        fit = max_route_len(sol)
        population.append((sol, fit))
        if fit < best_fitness:
            best_fitness = fit
            best_solution = [r[:] for r in sol]
            report_best_vrp(best_solution)

    # Generations
    max_gen = 20 + n // 5
    for gen in range(max_gen):
        # Selection: tournament (size 3)
        new_population = []
        # Elitism: keep best 2
        population.sort(key=lambda x: x[1])
        new_population.append(([r[:] for r in population[0][0]], population[0][1]))
        if len(population) > 1:
            new_population.append(([r[:] for r in population[1][0]], population[1][1]))
        while len(new_population) < pop_size:
            # tournament
            tournament = random.sample(population, min(3, len(population)))
            tournament.sort(key=lambda x: x[1])
            parent1 = [r[:] for r in tournament[0][0]]
            tournament = random.sample(population, min(3, len(population)))
            tournament.sort(key=lambda x: x[1])
            parent2 = [r[:] for r in tournament[0][0]]
            # crossover
            child = crossover(parent1, parent2)
            # mutation
            if random.random() < 0.1:
                child = mutate(child)
            fit = max_route_len(child)
            new_population.append((child, fit))
            if fit < best_fitness:
                best_fitness = fit
                best_solution = [r[:] for r in child]
                report_best_vrp(best_solution)
        population = new_population

    # Ensure exactly truck_count routes and each customer appears once (should be fine)
    # Check feasibility and return best
    if best_solution is None:
        best_solution = population[0][0]
    return best_solution