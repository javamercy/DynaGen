import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    dist = distance_matrix
    best_tour = None
    best_dist = float('inf')
    restarts = max(1, int(np.log2(n)))
    for _ in range(restarts):
        # Nearest neighbor from random start
        start = random.randrange(n)
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        curr = start
        while unvisited:
            next_city = min(unvisited, key=lambda c: dist[curr, c])
            tour.append(next_city)
            unvisited.remove(next_city)
            curr = next_city
        cur_dist = sum(dist[tour[i]][tour[(i+1)%n]] for i in range(n))
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)
        # 2-opt local search
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a, b = tour[i], tour[(i+1)%n]
                    c, d = tour[j], tour[(j+1)%n]
                    delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                    if delta < -1e-12:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        cur_dist += delta
                        if cur_dist < best_dist - 1e-12:
                            best_dist = cur_dist
                            best_tour = np.array(tour)
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
    return best_tour