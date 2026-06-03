import numpy as np
def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n, dtype=int)
    # Nearest neighbor construction
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        next_node = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    tour = np.array(tour, dtype=int)
    report_best_tour(tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):  # skip i+1 to avoid reversing a single edge
                a = tour[i]
                b = tour[(i + 1) % n]
                c = tour[j]
                d = tour[(j + 1) % n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-9:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
                    report_best_tour(tour)
    return tour