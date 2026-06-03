import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    # nearest neighbor initial tour
    tour = [0]
    unvisited = set(range(1, n))
    cur = 0
    while unvisited:
        next_city = min(unvisited, key=lambda x: distance_matrix[cur, x])
        tour.append(next_city)
        unvisited.remove(next_city)
        cur = next_city
    best_tour = np.array(tour)
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    
    pop_size = min(100, max(10, n))
    population = [best_tour.copy()]
    for _ in range(pop_size - 1):
        perm = np.random.permutation(n)
        population.append(perm)
    
    def dist(tour):
        total = 0
        for i in range(n-1):
            total += distance_matrix[tour[i], tour[i+1]]
        total += distance_matrix[tour[-1], tour[0]]
        return total
    fitness = np.array([dist(t) for t in population])
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < best_dist:
        best_dist = fitness[best_idx]
        best_tour = population[best_idx].copy()
        report_best_tour(best_tour)
    
    def order_crossover(p1, p2):
        size = len(p1)
        a, b = sorted(np.random.choice(size, 2, replace=False))
        child1 = [-1]*size
        child1[a:b+1] = p1[a:b+1].tolist()
        pos = b+1
        for city in np.concatenate([p2[b+1:], p2[:b+1]]):
            if city not in child1:
                child1[pos % size] = city
                pos += 1
        child2 = [-1]*size
        child2[a:b+1] = p2[a:b+1].tolist()
        pos = b+1
        for city in np.concatenate([p1[b+1:], p1[:b+1]]):
            if city not in child2:
                child2[pos % size] = city
                pos += 1
        return np.array(child1), np.array(child2)
    
    def mutate(tour):
        i, j = np.random.choice(n, 2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]
        return tour
    
    generations = 1000
    elite_size = 2
    for gen in range(generations):
        new_pop = []
        elite_indices = np.argsort(fitness)[:elite_size]
        for idx in elite_indices:
            new_pop.append(population[idx].copy())
        while len(new_pop) < pop_size:
            idx1 = np.random.randint(pop_size)
            idx2 = np.random.randint(pop_size)
            p1 = population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]
            idx1 = np.random.randint(pop_size)
            idx2 = np.random.randint(pop_size)
            p2 = population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]
            if np.random.rand() < 0.8:
                c1, c2 = order_crossover(p1, p2)
                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    new_pop.append(c2)
            else:
                new_pop.append(p1.copy())
                if len(new_pop) < pop_size:
                    new_pop.append(p2.copy())
        for i in range(elite_size, pop_size):
            if np.random.rand() < 0.1:
                new_pop[i] = mutate(new_pop[i])
        population = new_pop
        fitness = np.array([dist(t) for t in population])
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_dist:
            best_dist = fitness[min_idx]
            best_tour = population[min_idx].copy()
            report_best_tour(best_tour)
    return best_tour