import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    current_tour = np.random.permutation(n)
    current_dist = np.sum(distance_matrix[current_tour[i], current_tour[(i+1)%n]] for i in range(n))
    best_tour = current_tour.copy()
    best_dist = current_dist
    report_best_tour(best_tour)
    
    max_dist = np.max(distance_matrix)
    initial_temp = max_dist * 0.1
    temp = initial_temp
    alpha = 0.995
    num_iterations = n * 100
    
    while temp > 1e-5:
        for _ in range(num_iterations):
            i = np.random.randint(0, n-2)
            j = np.random.randint(i+2, n)
            a, b, c, d = current_tour[i], current_tour[i+1], current_tour[j], current_tour[(j+1)%n]
            delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                current_tour[i+1:j+1] = current_tour[j:i:-1]
                current_dist += delta
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_tour = current_tour.copy()
                    report_best_tour(best_tour)
        temp *= alpha
    return best_tour