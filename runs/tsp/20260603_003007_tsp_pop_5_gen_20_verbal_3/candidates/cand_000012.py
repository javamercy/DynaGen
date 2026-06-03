import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour

    def total_dist(tour):
        total = 0
        for i in range(n - 1):
            total += distance_matrix[tour[i], tour[i+1]]
        total += distance_matrix[tour[-1], tour[0]]
        return total

    # initial nearest neighbor tour
    def nearest_neighbor():
        tour = np.empty(n, dtype=np.int32)
        unvisited = np.ones(n, dtype=bool)
        tour[0] = 0
        unvisited[0] = False
        current = 0
        for i in range(1, n):
            dists = np.where(unvisited, distance_matrix[current], np.inf)
            next_node = np.argmin(dists)
            tour[i] = next_node
            unvisited[next_node] = False
            current = next_node
        return tour

    best_tour = nearest_neighbor()
    best_dist = total_dist(best_tour)
    report_best_tour(best_tour.copy())

    pop_size = min(50, max(10, n))
    population = [best_tour.copy()]
    for _ in range(pop_size - 1):
        population.append(np.random.permutation(n))
    fitness = np.array([total_dist(t) for t in population])

    def update_best(tour):
        nonlocal best_dist, best_tour
        dist = total_dist(tour)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

    def order_crossover(p1, p2):
        size = n
        a, b = sorted(np.random.choice(size, 2, replace=False))
        child1 = [-1] * size
        child1[a:b+1] = p1[a:b+1].tolist()
        pos = b + 1
        for city in np.concatenate([p2[b+1:], p2[:b+1]]):
            if city not in child1:
                child1[pos % size] = city
                pos += 1
        child2 = [-1] * size
        child2[a:b+1] = p2[a:b+1].tolist()
        pos = b + 1
        for city in np.concatenate([p1[b+1:], p1[:b+1]]):
            if city not in child2:
                child2[pos % size] = city
                pos += 1
        return np.array(child1, dtype=np.int32), np.array(child2, dtype=np.int32)

    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 2, n):
                    delta = (distance_matrix[tour[i], tour[j]] +
                             distance_matrix[tour[(i+1)%n], tour[(j+1)%n]] -
                             distance_matrix[tour[i], tour[(i+1)%n]] -
                             distance_matrix[tour[j], tour[(j+1)%n]])
                    if delta < -1e-8:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        update_best(tour)
                        break
                if improved:
                    break
        return tour

    def mutate(tour):
        i, j = np.random.choice(n, 2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]
        update_best(tour)
        return tour

    generations = 200
    elite_size = 2
    for _ in range(generations):
        new_pop = []
        elite_indices = np.argsort(fitness)[:elite_size]
        for idx in elite_indices:
            new_pop.append(population[idx].copy())
        while len(new_pop) < pop_size:
            # tournament selection
            i1 = np.random.randint(pop_size)
            i2 = np.random.randint(pop_size)
            p1 = population[i1] if fitness[i1] < fitness[i2] else population[i2]
            i1 = np.random.randint(pop_size)
            i2 = np.random.randint(pop_size)
            p2 = population[i1] if fitness[i1] < fitness[i2] else population[i2]
            if random.random() < 0.8:
                c1, c2 = order_crossover(p1, p2)
                c1 = two_opt(c1)
                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    c2 = two_opt(c2)
                    new_pop.append(c2)
            else:
                new_pop.append(p1.copy())
                if len(new_pop) < pop_size:
                    new_pop.append(p2.copy())
        for i in range(elite_size, pop_size):
            if random.random() < 0.1:
                new_pop[i] = mutate(new_pop[i])
        population = new_pop
        fitness = np.array([total_dist(t) for t in population])
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_dist:
            best_dist = fitness[min_idx]
            best_tour = population[min_idx].copy()
            report_best_tour(best_tour)
    return best_tour