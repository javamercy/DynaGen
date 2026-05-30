import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    pop_size = 100
    generations = 500
    mutation_rate = 0.1
    elite_size = 5

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

    # Generate initial population
    pop = []
    for i in range(pop_size):
        if i < 5:  # some nearest neighbor seeds
            pop.append(nearest_neighbor(i % n))
        else:
            pop.append(np.random.permutation(n))
    pop = np.array(pop)

    best_tour = None
    best_dist = float('inf')

    def tour_distance(tour):
        return distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])

    def evaluate_population():
        nonlocal best_tour, best_dist
        for tour in pop:
            d = tour_distance(tour)
            if d < best_dist:
                best_dist = d
                best_tour = tour.copy()
                report_best_tour(best_tour)  # assume exists

    def tournament_selection(k=3):
        indices = random.sample(range(pop_size), k)
        best_idx = min(indices, key=lambda i: tour_distance(pop[i]))
        return pop[best_idx].copy()

    def edge_recombination_crossover(parent1, parent2):
        # Build adjacency lists
        adj = {i: set() for i in range(n)}
        for parent in [parent1, parent2]:
            for i in range(n):
                a = parent[i]
                b = parent[(i+1)%n]
                adj[a].add(b)
                adj[b].add(a)
        # Start with random city
        current = np.random.randint(n)
        child = [current]
        remaining = set(range(n))
        remaining.remove(current)
        while remaining:
            # Remove current from adjacency lists
            for city in adj:
                adj[city].discard(current)
            # Find neighbors in remaining
            neighbors = [c for c in adj[current] if c in remaining]
            if neighbors:
                # Choose neighbor with fewest remaining neighbors (if tie, random)
                min_degree = min(len(adj[c] & remaining) for c in neighbors)
                candidates = [c for c in neighbors if len(adj[c] & remaining) == min_degree]
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

    evaluate_population()

    for gen in range(generations):
        new_pop = []
        # Elitism
        sorted_indices = sorted(range(pop_size), key=lambda i: tour_distance(pop[i]))
        for idx in sorted_indices[:elite_size]:
            new_pop.append(pop[idx].copy())
        # Fill rest
        while len(new_pop) < pop_size:
            p1 = tournament_selection()
            p2 = tournament_selection()
            if random.random() < 0.8:
                child = edge_recombination_crossover(p1, p2)
            else:
                child = p1.copy()
            if random.random() < mutation_rate:
                child = swap_mutation(child)
            new_pop.append(child)
        pop = np.array(new_pop)
        evaluate_population()
        # Optional: early break if converged? Not necessary.

    return best_tour