import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    best_tour = None
    best_dist = np.inf
    max_restarts = 5
    for restart in range(max_restarts):
        # Nearest neighbor from random start
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            next_city = min(unvisited, key=lambda x: distance_matrix[current, x])
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        tour = np.array(tour, dtype=int)
        # If first restart, set as best and report
        if restart == 0:
            dist = distance_matrix[tour[-1], tour[0]]
            for k in range(n-1):
                dist += distance_matrix[tour[k], tour[k+1]]
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n):
                    a, b = tour[i], tour[(i+1) % n]
                    c, d = tour[j], tour[(j+1) % n]
                    delta = (distance_matrix[a, c] + distance_matrix[b, d]
                             - distance_matrix[a, b] - distance_matrix[c, d])
                    if delta < -1e-12:
                        tour[i+1:j+1] = np.flip(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
        # Evaluate distance after improvement
        dist = distance_matrix[tour[-1], tour[0]]
        for k in range(n-1):
            dist += distance_matrix[tour[k], tour[k+1]]
        if dist < best_dist - 1e-12:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour