import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n)

    best_tour = None
    best_dist = float('inf')

    for _ in range(10):
        # Build initial tour with two farthest cities
        max_dist = -1
        start = 0
        end = 1
        for i in range(n):
            for j in range(i+1, n):
                if distance_matrix[i, j] > max_dist:
                    max_dist = distance_matrix[i, j]
                    start, end = i, j
        tour = [start, end]
        in_tour = {start, end}

        # Randomized farthest insertion
        while len(tour) < n:
            candidates = []
            weights = []
            for city in range(n):
                if city in in_tour:
                    continue
                min_dist = min(distance_matrix[city, tour[k]] for k in range(len(tour)))
                candidates.append(city)
                weights.append(min_dist ** 2 + 1e-6)
            # Choose city with probability proportional to weight
            total = sum(weights)
            r = random.random() * total
            cum = 0
            for idx, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen = candidates[idx]
                    break
            # Insert at best position
            best_pos = 0
            best_inc = float('inf')
            for pos in range(len(tour)):
                prev = tour[pos]
                nxt = tour[(pos+1) % len(tour)]
                inc = distance_matrix[prev, chosen] + distance_matrix[chosen, nxt] - distance_matrix[prev, nxt]
                if inc < best_inc:
                    best_inc = inc
                    best_pos = pos+1
            tour.insert(best_pos, chosen)
            in_tour.add(chosen)

        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                    if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break

        # Evaluate
        tour_arr = np.array(tour)
        dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if dist < best_dist:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)

    return best_tour