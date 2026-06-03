import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    best_tour = None
    best_dist = float('inf')
    
    def tour_distance(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    
    for _ in range(20):  # number of restarts
        tour = np.random.permutation(n)
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+1, n):
                    if j - i == 1:
                        continue
                    a, b = tour[i], tour[(i+1)%n]
                    c, d = tour[j], tour[(j+1)%n]
                    if distance_matrix[a,c] + distance_matrix[b,d] < distance_matrix[a,b] + distance_matrix[c,d]:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
        dist = tour_distance(tour)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour