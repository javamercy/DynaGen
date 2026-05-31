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

    def create_initial_solution():
        customers = list(range(1, n))
        random.shuffle(customers)
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in customers:
            best_truck = None
            best_pos = None
            best_max = float('inf')
            best_total = float('inf')
            for t, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_routes = routes[:t] + [new_route] + routes[t+1:]
                    new_max = max(route_distance(r) for r in new_routes)
                    new_total = sum(route_distance(r) for r in new_routes)
                    if new_max < best_max or (new_max == best_max and new_total < best_total):
                        best_max = new_max
                        best_total = new_total
                        best_truck = t
                        best_pos = pos
            routes[best_truck].insert(best_pos, cust)
        return routes

    def greedy_repair(routes, unassigned):
        routes = [list(r) for r in routes]
        unassigned = list(unassigned)
        for cust in unassigned:
            best_max = float('inf')
            best_total = float('inf')
            best_truck = None
            best_pos = None
            for t, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_routes = routes[:t] + [new_route] + routes[t+1:]
                    new_max_val = max(route_distance(r) for r in new_routes)
                    new_total_val = sum(route_distance(r) for r in new_routes)
                    if new_max_val < best_max or (new_max_val == best_max and new_total_val < best_total):
                        best_max = new_max_val
                        best_total = new_total_val
                        best_truck = t
                        best_pos = pos
            routes[best_truck].insert(best_pos, cust)
        return routes

    def regret2_repair(routes, unassigned):
        routes = [list(r) for r in routes]
        unassigned = list(unassigned)
        while unassigned:
            best_info = None
            for cust in unassigned:
                best_max = float('inf')
                best_total = float('inf')
                second_best_max = float('inf')
                second_best_total = float('inf')
                best_truck = None
                best_pos = None
                for t, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        new_routes = routes[:t] + [new_route] + routes[t+1:]
                        new_max_val = max(route_distance(r) for r in new_routes)
                        new_total_val = sum(route_distance(r) for r in new_routes)
                        if new_max_val < best_max or (new_max_val == best_max and new_total_val < best_total):
                            second_best_max = best_max
                            second_best_total = best_total
                            best_max = new_max_val
                            best_total = new_total_val
                            best_truck = t
                            best_pos = pos
                        elif new_max_val < second_best_max or (new_max_val == second_best_max and new_total_val < second_best_total):
                            second_best_max = new_max_val
                            second_best_total = new_total_val
                if second_best_max == float('inf'):
                    regret = float('inf')
                else:
                    regret = second_best_max - best_max
                if best_info is None or regret > best_info[0] or (regret == best_info[0] and (best_max < best_info[1] or (best_max == best_info[1] and cust < best_info[2]))):
                    best_info = (regret, best_max, cust, best_truck, best_pos)
            _, _, cust, best_truck, best_pos = best_info
            routes[best_truck].insert(best_pos, cust)
            unassigned.remove(cust)
        return routes

    # Initial population
    pop_size = min(20, max(5, n // 10))
    population = []
    for _ in range(pop_size):
        sol = create_initial_solution()
        fitness = max(route_distance(r) for r in sol)
        population.append((fitness, sol))
    population.sort(key=lambda x: x[0])
    best_fitness = population[0][0]
    best_solution = [list(r) for r in population[0][1]]
    report_best_vrp(best_solution)

    max_generations = min(50, max(5, n // 5))
    mutation_rate = 0.2
    removal_fraction = 0.15
    num_removals = max(1, int(removal_fraction * (n-1)))
    no_improve = 0

    for gen in range(max_generations):
        # Crossover: two parents via tournament
        parent1 = random.choice(population[:max(2, len(population)//2)])[1]
        parent2 = random.choice(population[:max(2, len(population)//2)])[1]

        # Route-based crossover
        child = [[] for _ in range(truck_count)]
        assigned_customers = set()
        for t in range(truck_count):
            if random.random() < 0.5:
                child[t] = list(parent1[t])
                for c in parent1[t][1:-1]:
                    assigned_customers.add(c)
            else:
                child[t] = [0, 0]
        unassigned = [c for c in range(1, n) if c not in assigned_customers]
        child = greedy_repair(child, unassigned)

        # Mutation
        if random.random() < mutation_rate:
            # Random removal
            all_cust = [c for r in child for c in r[1:-1]]
            if len(all_cust) > num_removals:
                random.shuffle(all_cust)
                removed = set(all_cust[:num_removals])
                partial = [[0] + [c for c in r[1:-1] if c not in removed] + [0] for r in child]
                repair_op = random.choice([0, 1])
                if repair_op == 0:
                    child = greedy_repair(partial, removed)
                else:
                    child = regret2_repair(partial, removed)

        # Evaluate child
        child_fitness = max(route_distance(r) for r in child)
        # Replace worst if better
        if child_fitness < population[-1][0]:
            population[-1] = (child_fitness, [list(r) for r in child])
            population.sort(key=lambda x: x[0])
            if child_fitness < best_fitness - 1e-9 or (abs(child_fitness - best_fitness) < 1e-9 and sum(route_distance(r) for r in child) < sum(route_distance(r) for r in best_solution)):
                best_fitness = child_fitness
                best_solution = [list(r) for r in child]
                report_best_vrp(best_solution)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        if no_improve >= 10:
            # Restart: replace worst solutions with new random ones
            for i in range(len(population)//2, len(population)):
                sol = create_initial_solution()
                fitness = max(route_distance(r) for r in sol)
                population[i] = (fitness, sol)
            population.sort(key=lambda x: x[0])
            no_improve = 0

    return best_solution