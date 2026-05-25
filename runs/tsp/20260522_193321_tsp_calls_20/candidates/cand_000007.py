import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = list(range(n))
        try:
            report_best_tour(np.array(tour))
        except:
            pass
        return np.array(tour)
    random.seed(seed)
    max_restarts = min(10, max(1, budget // (n * 5)))
    total_2opt_iters = min(budget // 10, 1000)
    per_restart_iters = max(1, total_2opt_iters // max_restarts) if max_restarts > 0 else 0
    best_tour = None
    best_dist = float('inf')
    for restart in range(max_restarts):
        # Nearest neighbor construction
        start = random.randrange(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            nearest = None
            min_dist = float('inf')
            for city in unvisited:
                d = distance_matrix[current][city]
                if d < min_dist:
                    min_dist = d
                    nearest = city
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        # Compute initial distance
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour[i]][tour[(i+1)%n]]
        if dist < best_dist:
            best_dist = dist
            best_tour = list(tour)
            try:
                report_best_tour(np.array(tour))
            except:
                pass
        # 2-opt improvement
        iteration = 0
        improvement = True
        while improvement and iteration < per_restart_iters:
            improvement = False
            for i in range(n-1):
                for j in range(i+2, n):
                    if j - i == 1:
                        continue
                    a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
                    if distance_matrix[a][c] + distance_matrix[b][d] < distance_matrix[a][b] + distance_matrix[c][d]:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improvement = True
                        dist = 0.0
                        for k in range(n):
                            dist += distance_matrix[tour[k]][tour[(k+1)%n]]
                        if dist < best_dist:
                            best_dist = dist
                            best_tour = list(tour)
                            try:
                                report_best_tour(np.array(tour))
                            except:
                                pass
            iteration += 1
    return np.array(best_tour)