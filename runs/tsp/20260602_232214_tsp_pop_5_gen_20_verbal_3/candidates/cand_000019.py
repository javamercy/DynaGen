import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def compute_distance(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))

    def two_opt(tour):
        best = tour.copy()
        best_dist = compute_distance(best)
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = best[i]
                    b = best[i+1]
                    c = best[j]
                    d = best[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        new = best.copy()
                        new[i+1:j+1] = best[j:i:-1]
                        new_dist = best_dist + delta
                        best = new
                        best_dist = new_dist
                        improved = True
                        report_best_tour(best)
        return best, best_dist

    best_tour = None
    best_dist = float('inf')
    for _ in range(10):  # number of restarts
        tour = np.random.permutation(n)
        dist = compute_distance(tour)
        if dist < best_dist:
            best_tour = tour.copy()
            best_dist = dist
            report_best_tour(best_tour)
        improved_tour, improved_dist = two_opt(tour)
        if improved_dist < best_dist:
            best_tour = improved_tour.copy()
            best_dist = improved_dist
            report_best_tour(best_tour)
    return best_tour