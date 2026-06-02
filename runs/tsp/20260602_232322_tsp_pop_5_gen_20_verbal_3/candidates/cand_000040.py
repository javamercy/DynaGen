import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    dist = distance_matrix
    # Nearest neighbor construction
    start = 0
    unvisited = set(range(1, n))
    tour = [start]
    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda x: dist[current, x])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    tour = np.array(tour)
    # Compute initial distance
    current_dist = sum(dist[tour[i], tour[(i+1)%n]] for i in range(n))
    best_tour = tour.copy()
    best_dist = current_dist
    report_best_tour(best_tour)
    # 2-opt local search
    improved = True
    max_iter = 100
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(n-1):
            for j in range(i+2, n):
                if j == n-1 and i == 0:
                    continue
                # Current edges: (i, i+1) and (j, j+1)
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -1e-10:
                    # reverse segment i+1 to j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    current_dist += delta
                    if current_dist < best_dist - 1e-10:
                        best_dist = current_dist
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    improved = True
        # If improved, restart loop
    return best_tour