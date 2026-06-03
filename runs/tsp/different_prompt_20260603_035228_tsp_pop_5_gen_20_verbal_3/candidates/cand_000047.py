import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 1:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    best_total = float('inf')
    best_tour = None
    for _ in range(5):
        # Randomized nearest neighbor construction
        start = random.randrange(n)
        tour = [start]
        visited = {start}
        for _ in range(n - 1):
            last = tour[-1]
            # Candidate list: nearest 3 unvisited
            candidates = []
            for j in range(n):
                if j not in visited:
                    candidates.append((distance_matrix[last, j], j))
            candidates.sort(key=lambda x: x[0])
            # Pick randomly from top 3
            k = min(3, len(candidates))
            selected = random.choice(candidates[:k])
            tour.append(selected[1])
            visited.add(selected[1])
        total = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        # 2-opt improvement (steepest descent)
        improved = True
        while improved:
            improved = False
            best_i = best_j = -1
            best_delta = 0
            for i in range(n-2):
                for j in range(i+2, n):
                    if j == n-1:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[0]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[0]])
                    else:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[j+1]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[j+1]])
                    if delta > best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            if best_delta > 0:
                i, j = best_i, best_j
                tour[i+1:j+1] = reversed(tour[i+1:j+1])
                total -= best_delta
                improved = True
        if total < best_total - 1e-10:
            best_total = total
            best_tour = tour[:]
            report_best_tour(np.array(best_tour))
    # Return the best tour found
    return np.array(best_tour)