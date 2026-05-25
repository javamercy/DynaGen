import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour
    rng = random.Random(seed)
    candidate_size = 30 if n < 80 else 20
    candidates = []
    for i in range(n):
        dists = distance_matrix[i]
        order = np.argsort(dists)
        order = order[order != i][:candidate_size]
        candidates.append(set(order))
    start = rng.randrange(n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    while unvisited:
        best_city = None
        best_regret = -1.0
        best_pos = None
        best_cost = None
        for city in unvisited:
            costs = []
            for idx in range(len(tour)):
                if idx == len(tour) - 1:
                    cost = distance_matrix[tour[idx], city] + distance_matrix[city, tour[0]] - distance_matrix[tour[idx], tour[0]]
                else:
                    cost = distance_matrix[tour[idx], city] + distance_matrix[city, tour[idx+1]] - distance_matrix[tour[idx], tour[idx+1]]
                costs.append(cost)
            costs.sort()
            min_cost = costs[0]
            if len(costs) >= 3:
                regret = (costs[1] - min_cost) + (costs[2] - min_cost)
            elif len(costs) == 2:
                regret = costs[1] - min_cost
            else:
                regret = 0.0
            if regret > best_regret:
                best_regret = regret
                best_city = city
                best_pos = costs.index(min_cost)
                best_cost = min_cost
        tour.insert(best_pos + 1, best_city)
        unvisited.remove(best_city)
    tour = np.array(tour, dtype=int)
    best_tour = tour.copy()
    best_dist = _tour_length(tour, distance_matrix)
    report_best_tour(best_tour)
    pos = np.empty(n, dtype=int)
    for idx, node in enumerate(tour):
        pos[node] = idx
    evals = 0
    improved = True
    while evals < budget and improved:
        improved = False
        pairs = []
        for i in range(n):
            a = tour[i]
            for b in candidates[a]:
                j = pos[b]
                if j <= i:
                    continue
                if (i + 1) % n == j or (j + 1) % n == i:
                    continue
                pairs.append((i, j))
        rng.shuffle(pairs)
        for i, j in pairs:
            if evals >= budget:
                break
            a = tour[i]
            b = tour[(i+1)%n]
            c = tour[j]
            d = tour[(j+1)%n]
            old = distance_matrix[a,b] + distance_matrix[c,d]
            new = distance_matrix[a,c] + distance_matrix[b,d]
            delta = new - old
            evals += 1
            if delta < -1e-12:
                if i < j:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                else:
                    segment = np.concatenate((tour[i+1:], tour[:j+1]))
                    segment = segment[::-1]
                    tour[i+1:] = segment[:n-i-1]
                    tour[:j+1] = segment[n-i-1:]
                for idx in range(n):
                    pos[tour[idx]] = idx
                best_dist += delta
                if best_dist > _tour_length(tour, distance_matrix):
                    best_dist = _tour_length(tour, distance_matrix)
                best_tour = tour.copy()
                report_best_tour(best_tour)
                improved = True
                break
    return best_tour

def _tour_length(tour, dist):
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dist[tour[i], tour[(i+1)%n]]
    return total