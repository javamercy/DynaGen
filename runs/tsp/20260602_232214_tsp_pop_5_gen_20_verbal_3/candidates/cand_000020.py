import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = float('inf')
    num_restarts = max(10, n * 2)
    for _ in range(num_restarts):
        tour = np.random.permutation(n)
        dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        tour[i+1:j+1] = tour[j:i:-1]
                        dist += delta
                        improved = True
            if dist < best_dist:
                best_dist = dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
    return best_tour