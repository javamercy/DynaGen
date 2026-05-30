import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        report_best_tour(np.array([0]))
        return np.array([0])
    
    # Nearest neighbor construction
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        nearest = min(unvisited, key=lambda city: distance_matrix[current, city])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    best_tour = np.array(tour)
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                a = best_tour[i]
                b = best_tour[(i+1)%n]
                c = best_tour[j]
                d = best_tour[(j+1)%n]
                old = distance_matrix[a, b] + distance_matrix[c, d]
                new = distance_matrix[a, c] + distance_matrix[b, d]
                if new < old:
                    # Reverse segment i+1..j
                    best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                    best_dist = best_dist - old + new
                    report_best_tour(best_tour)
                    improved = True
    return best_tour