import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    # Parameters
    pop_size = 20
    generations = 100
    elitism = 1
    crossover_rate = 0.8
    mutation_rate = 0.1
    
    def compute_dist(tour):
        s = 0.0
        for i in range(n):
            s += distance_matrix[tour[i], tour[(i+1)%n]]
        return s
    
    # Initialize population
    population = []
    # Nearest neighbor from random start
    start = random.randrange(n)
    tour = [start]
    visited = {start}
    current = start
    for _ in range(n-1):
        next_node = min([j for j in range(n) if j not in visited], key=lambda j: distance_matrix[current, j])
        tour.append(next_node)
        visited.add(next_node)
        current = next_node
    population.append(np.array(tour))
    # Fill with random permutations
    for _ in range(pop_size - 1):
        perm = list(range(n))
        random.shuffle(perm)
        population.append(np.array(perm))
    
    # Evaluate initial fitness
    fitness = [compute_dist(tour) for tour in population]
    best_idx = np.argmin(fitness)
    best_tour = population[best_idx].copy()
    best_dist = fitness[best_idx]
    report_best_tour(best_tour)
    
    # Evolutionary loop
    for gen in range(generations):
        new_population = []
        # Elitism
        elite_indices = np.argsort(fitness)[:elitism]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament():
                i = random.randrange(pop_size)
                j = random.randrange(pop_size)
                return population[i] if fitness[i] < fitness[j] else population[j]
            parent1 = tournament()
            parent2 = tournament()
            if random.random() < crossover_rate:
                # Order crossover (OX1)
                a, b = sorted(random.sample(range(n), 2))
                child = np.full(n, -1)
                child[a:b+1] = parent1[a:b+1]
                pos = (b+1) % n
                for city in parent2:
                    if city not in child:
                        child[pos] = city
                        pos = (pos + 1) % n
                child1 = child
                # second child
                a, b = sorted(random.sample(range(n), 2))
                child = np.full(n, -1)
                child[a:b+1] = parent2[a:b+1]
                pos = (b+1) % n
                for city in parent1:
                    if city not in child:
                        child[pos] = city
                        pos = (pos + 1) % n
                child2 = child
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            # Mutation
            if random.random() < mutation_rate:
                i, j = random.sample(range(n), 2)
                child1[i], child1[j] = child1[j], child1[i]
            if random.random() < mutation_rate:
                i, j = random.sample(range(n), 2)
                child2[i], child2[j] = child2[j], child2[i]
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = new_population
        fitness = [compute_dist(tour) for tour in population]
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_dist - 1e-10:
            best_dist = fitness[current_best_idx]
            best_tour = population[current_best_idx].copy()
            report_best_tour(best_tour)
    
    return best_tour