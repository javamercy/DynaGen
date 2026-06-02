import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # Nearest neighbor construction
    start = 0
    tour = [start]
    unvisited = set(range(1, n))
    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda city: distance_matrix[current, city])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    # Compute initial distance
    tour_arr = np.array(tour)
    dist = 0.0
    for i in range(n):
        dist += distance_matrix[tour_arr[i], tour_arr[(i+1)%n]]
    best_dist = dist
    best_tour = tour_arr.copy()
    report_best_tour(best_tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
                delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                if delta < -1e-12:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    dist += delta
                    if dist < best_dist - 1e-12:
                        best_dist = dist
                        best_tour = np.array(tour)
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved:
                break
    return best_tour