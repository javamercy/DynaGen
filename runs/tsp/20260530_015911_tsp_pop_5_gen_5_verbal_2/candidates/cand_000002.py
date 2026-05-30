import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    # Nearest neighbor initial tour
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    tour = np.array(tour, dtype=int)
    # report initial tour
    report_best_tour(tour.copy())
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                # edges (i,i+1) and (j,(j+1)%n)
                a, b = tour[i], tour[(i+1) % n]
                c, d = tour[j], tour[(j+1) % n]
                delta = (distance_matrix[a, c] + distance_matrix[b, d]
                         - distance_matrix[a, b] - distance_matrix[c, d])
                if delta < -1e-12:  # improvement
                    # reverse segment i+1..j
                    tour[i+1:j+1] = np.flip(tour[i+1:j+1])
                    improved = True
                    report_best_tour(tour.copy())
                    break
            if improved:
                break
    return tour