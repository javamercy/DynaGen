import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = len(distance_matrix)
    rng = np.random.default_rng(seed)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        rng.shuffle(tour)
        report_best_tour(tour)
        return tour

    def total_distance(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))

    pop_size = min(20, max(5, budget // 10))
    if pop_size < 5:
        pop_size = 5
    if budget < pop_size:
        tour = rng.permutation(n).astype(np.int64)
        report_best_tour(tour)
        return tour

    population = [rng.permutation(n).astype(np.int64) for _ in range(pop_size)]
    distances = [total_distance(t) for t in population]
    best_tour = population[np.argmin(distances)].copy()
    best_dist = np.min(distances)
    report_best_tour(best_tour)
    evals = pop_size

    def tournament_select(pop, fits, k, rng):
        idx = rng.integers(len(pop), size=k)
        best_idx = idx[np.argmin([fits[i] for i in idx])]
        return pop[best_idx].copy()

    def order_crossover(p1, p2, rng):
        n = len(p1)
        a, b = sorted(rng.integers(0, n, size=2))
        child1 = np.full(n, -1, dtype=np.int64)
        child1[a:b] = p1[a:b]
        ptr = b
        for city in p2:
            if city not in child1:
                child1[ptr % n] = city
                ptr += 1
        child2 = np.full(n, -1, dtype=np.int64)
        child2[a:b] = p2[a:b]
        ptr = b
        for city in p1:
            if city not in child2:
                child2[ptr % n] = city
                ptr += 1
        return child1, child2

    def mutate(tour, rng, prob=0.1):
        if rng.random() < prob:
            i, j = rng.integers(n, size=2)
            tour[i], tour[j] = tour[j], tour[i]

    while evals < budget:
        p1 = tournament_select(population, distances, 3, rng)
        p2 = tournament_select(population, distances, 3, rng)
        c1, c2 = order_crossover(p1, p2, rng)
        mutate(c1, rng)
        mutate(c2, rng)
        d1 = total_distance(c1)
        d2 = total_distance(c2)
        evals += 2
        if d1 < best_dist:
            best_dist = d1
            best_tour = c1.copy()
            report_best_tour(best_tour)
        if d2 < best_dist:
            best_dist = d2
            best_tour = c2.copy()
            report_best_tour(best_tour)
        worst_idx = np.argmax(distances)
        if d1 < distances[worst_idx]:
            population[worst_idx] = c1
            distances[worst_idx] = d1
        worst_idx2 = np.argmax(distances)
        if d2 < distances[worst_idx2]:
            population[worst_idx2] = c2
            distances[worst_idx2] = d2
        if evals >= budget:
            break

    return best_tour