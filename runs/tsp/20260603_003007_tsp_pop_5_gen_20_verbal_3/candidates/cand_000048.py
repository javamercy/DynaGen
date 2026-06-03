import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour

    def total_dist(tour):
        total = 0.0
        for i in range(n - 1):
            total += distance_matrix[tour[i], tour[i+1]]
        total += distance_matrix[tour[-1], tour[0]]
        return total

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

    pop_size = min(20, n)
    population = [best_tour.copy()]
    for _ in range(pop_size - 1):
        population.append(np.random.permutation(n))
    fitness = np.array([total_dist(t) for t in population])

    def update_best(tour):
        nonlocal best_dist, best_tour
        d = total_dist(tour)
        if d < best_dist:
            best_dist = d
            best_tour = tour.copy()
            report_best_tour(best_tour)

    def order_crossover(p1, p2):
        a, b = sorted(np.random.choice(n, 2, replace=False))
        child = np.full(n, -1, dtype=np.int32)
        child[a:b+1] = p1[a:b+1]
        pos = (b + 1) % n
        for city in p2:
            if city not in child:
                child[pos] = city
                pos = (pos + 1) % n
        return child

    def one_pass_2opt(tour):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                delta = (distance_matrix[tour[i], tour[j]] +
                         distance_matrix[tour[(i+1)%n], tour[(j+1)%n]] -
                         distance_matrix[tour[i], tour[(i+1)%n]] -
                         distance_matrix[tour[j], tour[(j+1)%n]])
                if delta < -1e-8:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    updated = tour.copy()
                    update_best(updated)
                    improved = True
                    break
            if improved:
                break
        return tour

    def mutate(tour):
        i, j = np.random.choice(n, 2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]
        update_best(tour)
        return tour

    generations = 50
    elite_size = 2
    for _ in range(generations):
        new_pop = []
        elite_indices = np.argsort(fitness)[:elite_size]
        for idx in elite_indices:
            new_pop.append(population[idx].copy())
        while len(new_pop) < pop_size:
            i1 = np.random.randint(pop_size)
            i2 = np.random.randint(pop_size)
            p1 = population[i1] if fitness[i1] < fitness[i2] else population[i2]
            i1 = np.random.randint(pop_size)
            i2 = np.random.randint(pop_size)
            p2 = population[i1] if fitness[i1] < fitness[i2] else population[i2]
            if random.random() < 0.8:
                child = order_crossover(p1, p2)
                child = one_pass_2opt(child)
                new_pop.append(child)
            else:
                new_pop.append(p1.copy())
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