import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 1:
        return np.array([0])
    
    # Nearest neighbor construction
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        nearest = None
        min_dist = np.inf
        for city in unvisited:
            d = distance_matrix[current, city]
            if d < min_dist:
                min_dist = d
                nearest = city
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    tour = np.array(tour)
    best_tour = tour.copy()
    best_length = np.sum(distance_matrix[tour[np.arange(n)], tour[(np.arange(n) + 1) % n]])
    report_best_tour(best_tour)
    
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                a, b = tour[i], tour[(i+1) % n]
                c, d = tour[j], tour[(j+1) % n]
                old = distance_matrix[a, b] + distance_matrix[c, d]
                new = distance_matrix[a, c] + distance_matrix[b, d]
                if new < old:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    new_length = best_length - old + new
                    if new_length < best_length:
                        best_length = new_length
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour