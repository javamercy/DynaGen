import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    best_tour = None
    best_dist = float('inf')
    total_passes = 0
    while total_passes < budget:
        # Nearest neighbor construction
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        last = start
        while unvisited:
            best_city = min(unvisited, key=lambda city: distance_matrix[last, city])
            tour.append(best_city)
            unvisited.remove(best_city)
            last = best_city
        tour = np.array(tour, dtype=np.int32)
        dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if dist < best_dist - 1e-12:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        # 2-opt improvement
        improved = True
        while improved and total_passes < budget:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        dist += delta
                        if dist < best_dist - 1e-12:
                            best_dist = dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
            total_passes += 1
    return best_tour