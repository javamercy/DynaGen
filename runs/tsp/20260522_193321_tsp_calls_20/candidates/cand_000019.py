import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=int)
        try:
            report_best_tour(tour)
        except:
            pass
        return tour

    rng = random.Random(seed)
    cand_size = 40
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        order = np.argsort(dists)
        order = order[order != i][:cand_size]
        candidates.append(set(order))

    def get_pos(t):
        pos = np.empty(n, dtype=int)
        for idx, node in enumerate(t):
            pos[node] = idx
        return pos

    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i+1)%n]]
        return total

    total_evals = 0
    best_tour = None
    best_dist = float('inf')

    # Regret-2 construction
    start = rng.randrange(n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    while unvisited:
        best_city = None
        best_regret = -1.0
        best_pos = None
        for city in unvisited:
            min_cost = float('inf')
            second_min = float('inf')
            best_idx = -1
            for idx in range(len(tour)):
                if idx == len(tour) - 1:
                    cost = distance_matrix[tour[idx], city] + distance_matrix[city, tour[0]] - distance_matrix[tour[idx], tour[0]]
                else:
                    cost = distance_matrix[tour[idx], city] + distance_matrix[city, tour[idx+1]] - distance_matrix[tour[idx], tour[idx+1]]
                if cost < min_cost:
                    second_min = min_cost
                    min_cost = cost
                    best_idx = idx
                elif cost < second_min:
                    second_min = cost
            regret = second_min - min_cost
            if regret > best_regret:
                best_regret = regret
                best_city = city
                best_pos = best_idx
        tour.insert(best_pos + 1, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour, dtype=int)
    current_dist = tour_length(tour)
    if current_dist < best_dist - 1e-12:
        best_dist = current_dist
        best_tour = tour.copy()
        try:
            report_best_tour(best_tour)
        except:
            pass

    pos = get_pos(tour)
    improved = True
    while improved and total_evals < budget:
        improved = False
        # Generate candidate pairs and shuffle
        pairs = []
        for i in range(n):
            a = tour[i]
            for b in candidates[a]:
                j = pos[b]
                if j <= i or (i+1)%n == j or (j+1)%n == i:
                    continue
                pairs.append((i, j))
        rng.shuffle(pairs)
        for i, j in pairs:
            if total_evals >= budget:
                break
            a = tour[i]
            b = tour[(i+1)%n]
            c = tour[j]
            d = tour[(j+1)%n]
            old = distance_matrix[a, b] + distance_matrix[c, d]
            new = distance_matrix[a, c] + distance_matrix[b, d]
            delta = new - old
            total_evals += 1
            if delta < -1e-12:
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
                pos = get_pos(tour)
                current_dist += delta
                if current_dist < best_dist - 1e-12:
                    best_dist = current_dist
                    best_tour = tour.copy()
                    try:
                        report_best_tour(best_tour)
                    except:
                        pass
                improved = True
                break
    return best_tour