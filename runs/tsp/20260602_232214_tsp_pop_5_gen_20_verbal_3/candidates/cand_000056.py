import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    rng = np.random.default_rng()
    pop_size = 30
    num_generations = 1000
    crossover_rate = 0.8
    mutation_rate = 0.1
    tournament_size = 3

    def tour_distance(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))

    def order_crossover(parent1, parent2):
        n = len(parent1)
        start, end = sorted(rng.integers(0, n, 2))
        child = [-1] * n
        child[start:end+1] = parent1[start:end+1].tolist()
        remaining = [gene for gene in parent2 if gene not in child]
        pos = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = remaining[pos]
                pos += 1
        return np.array(child)

    def inversion_mutation(tour):
        i, j = sorted(rng.integers(0, n, 2))
        tour[i:j+1] = tour[i:j+1][::-1]
        return tour

    pop = [rng.permutation(n) for _ in range(pop_size)]
    fitnesses = [tour_distance(tour) for tour in pop]
    best_idx = np.argmin(fitnesses)
    best_tour = pop[best_idx].copy()
    best_dist = fitnesses[best_idx]
    report_best_tour(best_tour)

    for gen in range(num_generations):
        new_pop = [best_tour.copy()]
        while len(new_pop) < pop_size:
            competitors = rng.integers(0, pop_size, tournament_size)
            winner1 = min(competitors, key=lambda i: fitnesses[i])
            competitors = rng.integers(0, pop_size, tournament_size)
            winner2 = min(competitors, key=lambda i: fitnesses[i])
            parent1 = pop[winner1]
            parent2 = pop[winner2]
            if rng.random() < crossover_rate:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1.copy()
            if rng.random() < mutation_rate:
                child = inversion_mutation(child)
            new_pop.append(child)
        pop = new_pop
        fitnesses = [tour_distance(tour) for tour in pop]
        gen_best_idx = np.argmin(fitnesses)
        gen_best_dist = fitnesses[gen_best_idx]
        if gen_best_dist < best_dist:
            best_dist = gen_best_dist
            best_tour = pop[gen_best_idx].copy()
            report_best_tour(best_tour)

    return best_tour