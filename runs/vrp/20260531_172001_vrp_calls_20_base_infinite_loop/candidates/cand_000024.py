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
    
    def decode(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        for cust in perm:
            best_max = float('inf')
            best_route_idx = None
            best_pos = None
            best_cost = None
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    nxt = route[pos]
                    cost = distance_matrix[prev, cust] + distance_matrix[cust, nxt] - distance_matrix[prev, nxt]
                    new_len = route_length(route) + cost
                    other_lens = [route_length(r) for i, r in enumerate(routes) if i != r_idx]
                    new_max = max(new_len, *other_lens)
                    if new_max < best_max or (new_max == best_max and (best_cost is None or cost > best_cost)):
                        best_max = new_max
                        best_route_idx = r_idx
                        best_pos = pos
                        best_cost = cost
            routes[best_route_idx].insert(best_pos, cust)
        max_len = max(route_length(r) for r in routes)
        return routes, max_len
    
    customers = list(range(1, n))
    perm_len = n - 1
    pop_size = min(20, max(5, n // 2))
    max_gen = min(50, n * 2)
    
    # Initialize population with random permutations
    pop = []
    for _ in range(pop_size):
        perm = customers[:]
        random.shuffle(perm)
        pop.append(perm)
    
    best_routes = None
    best_max = float('inf')
    
    def evaluate_and_update(perm):
        nonlocal best_routes, best_max
        routes, max_len = decode(perm)
        if max_len < best_max:
            best_max = max_len
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        return max_len
    
    # Evaluate initial population
    fitness = [evaluate_and_update(perm) for perm in pop]
    
    # Order crossover (OX)
    def order_crossover(parent1, parent2):
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child1 = [None]*size
        child2 = [None]*size
        # Copy segment from parent1 to child1
        child1[a:b+1] = parent1[a:b+1]
        # Fill remaining positions from parent2 in order, wrapping
        pointer = (b+1) % size
        for i in range(size):
            idx = (b+1 + i) % size
            if parent2[idx] not in child1:
                child1[pointer] = parent2[idx]
                pointer = (pointer + 1) % size
        # Same for child2
        child2[a:b+1] = parent2[a:b+1]
        pointer = (b+1) % size
        for i in range(size):
            idx = (b+1 + i) % size
            if parent1[idx] not in child2:
                child2[pointer] = parent1[idx]
                pointer = (pointer + 1) % size
        return child1, child2
    
    # Swap mutation
    def mutate(perm):
        if random.random() < 0.1:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
    
    # Main GA loop
    for gen in range(max_gen):
        # Selection (tournament size 3)
        selected = []
        for _ in range(pop_size):
            indices = random.sample(range(pop_size), 3)
            best_idx = indices[0]
            for idx in indices[1:]:
                if fitness[idx] < fitness[best_idx]:
                    best_idx = idx
            selected.append(pop[best_idx])
        
        # Generate offspring
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = selected[i]
            p2 = selected[i+1] if i+1 < pop_size else selected[0]
            c1, c2 = order_crossover(p1, p2)
            mutate(c1)
            mutate(c2)
            offspring.append(c1)
            offspring.append(c2)
        offspring = offspring[:pop_size]  # ensure size
        
        # Evaluate offspring
        offspring_fitness = [evaluate_and_update(perm) for perm in offspring]
        
        # Combine current and offspring, keep best pop_size
        combined = list(zip(pop, fitness)) + list(zip(offspring, offspring_fitness))
        combined.sort(key=lambda x: x[1])
        combined = combined[:pop_size]
        pop = [perm for perm, _ in combined]
        fitness = [fit for _, fit in combined]
    
    # Return best found
    return best_routes