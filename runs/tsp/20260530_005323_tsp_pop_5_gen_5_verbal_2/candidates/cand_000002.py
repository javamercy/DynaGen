import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 1:
        return np.array([0])
    
    # Cheapest insertion construction
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        best_increase = np.inf
        best_city = None
        best_pos = None
        for city in unvisited:
            for pos in range(len(tour) + 1):
                prev = tour[pos - 1] if pos > 0 else tour[-1]
                nxt = tour[pos] if pos < len(tour) else tour[0]
                increase = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                if increase < best_increase:
                    best_increase = increase
                    best_city = city
                    best_pos = pos
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    
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
                    # Reverse segment from i+1 to j
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