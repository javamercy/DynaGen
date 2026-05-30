import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    pop_size = 50
    generations = 300
    mutation_rate = 0.1
    crossover_rate = 0.8
    elite_size = 3
    ls_prob = 0.5

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

    pop = []
    for i in range(pop_size):
        if i < 10:
            pop.append(nearest_neighbor(random.randint(0, n-1)))
        else:
            pop.append(np.random.permutation(n))
    pop = np.array(pop)

    best_tour = None
    best_dist = float('inf')
    last_improvement_gen = 0

    def report_if_best(tour):
        nonlocal best_tour, best_dist, last_improvement_gen
        d = tour_distance(tour)
        if d < best_dist - 1e-10:
            best_dist = d
            best_tour = tour.copy()
            last_improvement_gen = gen
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

    def two_opt_local_search(tour):
        best = tour.copy()
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    if j == i+1:
                        continue
                    # Compute delta for reversing segment i+1..j
                    # Current edges: (i, i+1) and (j, (j+1)%n)
                    next_i = (i+1) % n
                    next_j = (j+1) % n
                    delta = (-distance_matrix[best[i], best[next_i]]
                             - distance_matrix[best[j], best[next_j]]
                             + distance_matrix[best[i], best[j]]
                             + distance_matrix[best[next_i], best[next_j]])
                    if delta < -1e-10:
                        # Reverse segment i+1..j
                        if j+1 < n:
                            best = np.concatenate([best[:i+1], best[i+1:j+1][::-1], best[j+1:]])
                        else:
                            best = np.concatenate([best[:i+1], best[i+1:][::-1]])
                        improved = True
                        break
                if improved:
                    break
        return best

    gen = 0
    for gen in range(generations):
        for tour in pop:
            report_if_best(tour)
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
                child = two_opt_local_search(child)
            new_pop.append(child)
        pop = np.array(new_pop)
        # Diversification: if no improvement for 50 generations, restart bottom half
        if gen - last_improvement_gen >= 50:
            for i in range(pop_size//2, pop_size):
                pop[i] = np.random.permutation(n)
            last_improvement_gen = gen

    # Final local search on best tour
    best_tour = two_opt_local_search(best_tour)
    report_if_best(best_tour)
    return best_tour