import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n <= 1:
        return [[0, 0] for _ in range(truck_count)]

    def route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in perm:
            best_new_max = float('inf')
            best_cost = float('inf')
            best_route_idx = -1
            best_pos = -1
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_len = route_length(new_route)
                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    if new_max < best_new_max or (new_max == best_new_max and new_len < best_cost):
                        best_new_max = new_max
                        best_cost = new_len
                        best_route_idx = r_idx
                        best_pos = pos
            routes[best_route_idx].insert(best_pos, cust)
        return routes, max(route_length(r) for r in routes)

    def order_crossover(p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None]*size
        child[a:b+1] = p1[a:b+1]
        remaining = [x for x in p2 if x not in child]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[idx]
                idx += 1
        return child

    def mutate(perm):
        perm = perm[:]
        if random.random() < 0.1:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        elif random.random() < 0.2:
            i, j = sorted(random.sample(range(len(perm)), 2))
            perm[i:j+1] = reversed(perm[i:j+1])
        return perm

    # Parameters
    pop_size = min(50, max(20, n))
    generations = min(100, n*5)
    tournament_size = 3
    elite_count = max(1, pop_size // 10)

    # Initialize population
    customers = list(range(1, n))
    pop = [random.sample(customers, len(customers)) for _ in range(pop_size)]
    # Include one greedy construction (min-max regret) but encode as permutation? Not needed for radical change.

    best_routes = None
    best_max = float('inf')

    for gen in range(generations):
        # Evaluate
        fitness = []
        for perm in pop:
            routes, max_len = decode(perm)
            fitness.append((max_len, perm, routes))
        fitness.sort(key=lambda x: x[0])
        if fitness[0][0] < best_max:
            best_max = fitness[0][0]
            best_routes = [r[:] for r in fitness[0][2]]
            report_best_vrp(best_routes)
        # Selection
        new_pop = []
        # Elitism
        for i in range(elite_count):
            new_pop.append(fitness[i][1])
        # Generate offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            t1 = random.sample(fitness, tournament_size)
            t1.sort(key=lambda x: x[0])
            p1 = t1[0][1]
            t2 = random.sample(fitness, tournament_size)
            t2.sort(key=lambda x: x[0])
            p2 = t2[0][1]
            child = order_crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        pop = new_pop

    # Fallback: if none found (should not happen), construct greedy
    if best_routes is None:
        routes = [[0, 0] for _ in range(truck_count)]
        unassigned = list(range(1, n))
        random.shuffle(unassigned)
        for cust in unassigned:
            best_new_max = float('inf')
            best_route_idx = -1
            best_pos = -1
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_len = route_length(new_route)
                    other_lens = [route_length(routes[i]) for i in range(truck_count) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
            routes[best_route_idx].insert(best_pos, cust)
        best_routes = routes
    return best_routes