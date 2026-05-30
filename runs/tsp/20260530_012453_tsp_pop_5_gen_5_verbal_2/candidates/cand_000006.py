import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    pop_size = 50
    generations = 200
    mutation_rate = 0.1
    crossover_rate = 0.8
    elite_size = 3
    ls_prob = 0.2  # probability of 2-opt improvement per child

    def tour_distance(tour):
        return distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])

    def nearest_neighbor(start):
        tour = [start]
        visited = {start}
        current = start
        for _ in range(n-1):
            next_city = min((i for i in range(n) if i not in visited), key=lambda x: distance_matrix[current, x])
            tour.append(next_city)
            visited.add(next_city)
            current = next_city
        return np.array(tour)

    # Initial population
    pop = []
    for i in range(pop_size):
        if i < 10:
            pop.append(nearest_neighbor(random.randint(0, n-1)))
        else:
            pop.append(np.random.permutation(n))
    pop = np.array(pop)

    best_tour = None
    best_dist = float('inf')

    def report_if_best(tour):
        nonlocal best_tour, best_dist
        d = tour_distance(tour)
        if d < best_dist:
            best_dist = d
            best_tour = tour.copy()
            report_best_tour(best_tour)

    for tour in pop:
        report_if_best(tour)

    def tournament_select(k=3):
        idx = min(random.sample(range(pop_size), k), key=lambda i: tour_distance(pop[i]))
        return pop[idx].copy()

    def edge_recombination_crossover(p1, p2):
        adj = {i: set() for i in range(n)}
        for parent in [p1, p2]:
            for i in range(n):
                a = parent[i]
                b = parent[(i+1)%n]
                adj[a].add(b)
                adj[b].add(a)
        current = np.random.randint(n)
        child = [current]
        remaining = set(range(n))
        remaining.remove(current)
        while remaining:
            for city in adj:
                adj[city].discard(current)
            neighbors = [c for c in adj[current] if c in remaining]
            if neighbors:
                min_deg = min(len(adj[c] & remaining) for c in neighbors)
                candidates = [c for c in neighbors if len(adj[c] & remaining) == min_deg]
                next_city = random.choice(candidates)
            else:
                next_city = random.choice(list(remaining))
            child.append(next_city)
            remaining.remove(next_city)
            current = next_city
        return np.array(child)

    def swap_mutation(tour):
        tour = tour.copy()
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]
        return tour

    def random_2opt_improve(tour):
        i, j = random.sample(range(n), 2)
        if i > j:
            i, j = j, i
        if j - i == 1 or (i == 0 and j == n-1):
            return tour
        delta = 0
        prev_i = tour[(i-1)%n]
        next_j = tour[(j+1)%n]
        delta -= distance_matrix[prev_i, tour[i]] + distance_matrix[tour[j], next_j]
        delta += distance_matrix[prev_i, tour[j]] + distance_matrix[tour[i], next_j]
        if delta < 0:
            new_tour = np.concatenate([tour[:i], tour[i:j+1][::-1], tour[j+1:]])
            return new_tour
        return tour

    for gen in range(generations):
        # Evaluate and keep best
        for tour in pop:
            report_if_best(tour)
        # Elitism
        sorted_idx = sorted(range(pop_size), key=lambda i: tour_distance(pop[i]))
        new_pop = [pop[i].copy() for i in sorted_idx[:elite_size]]
        while len(new_pop) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            if random.random() < crossover_rate:
                child = edge_recombination_crossover(p1, p2)
            else:
                child = p1.copy()
            if random.random() < mutation_rate:
                child = swap_mutation(child)
            if random.random() < ls_prob:
                child = random_2opt_improve(child)
            new_pop.append(child)
        pop = np.array(new_pop)

    # Final improvement on best tour
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                new_tour = random_2opt_improve(best_tour)  # but we need deterministic
                # Actually do a full 2-opt pass
                pass
    # Simpler: apply random 2-opt on best many times
    for _ in range(1000):
        best_tour = random_2opt_improve(best_tour)
    report_if_best(best_tour)
    return best_tour