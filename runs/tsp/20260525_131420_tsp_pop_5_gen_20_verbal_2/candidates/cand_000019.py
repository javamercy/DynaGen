import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    # Nearest neighbor construction
    start = np.random.randint(n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    last = start
    while unvisited:
        best_dist = float('inf')
        best_city = -1
        for city in unvisited:
            d = distance_matrix[last, city]
            if d < best_dist:
                best_dist = d
                best_city = city
        tour.append(best_city)
        unvisited.remove(best_city)
        last = best_city
    tour = np.array(tour, dtype=np.int32)
    total = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    best_tour = tour.copy()
    best_dist = total
    report_best_tour(best_tour)
    # 2-opt improvement
    iteration = 0
    improved = True
    while improved and iteration < budget:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                if iteration >= budget:
                    break
                a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    total += delta
                    if total < best_dist - 1e-12:
                        best_dist = total
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved or iteration >= budget:
                break
        iteration += 1
    return best_tour