import numpy as np

def total_distance(dist, tour):
    return sum(dist[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    # farthest insertion construction
    start, end = np.unravel_index(np.argmax(distance_matrix), (n, n))
    tour = [start, end]
    unvisited = set(range(n)) - {start, end}
    while unvisited:
        farthest_dist = -1
        farthest_city = -1
        for city in unvisited:
            dist = min(distance_matrix[city][t] for t in tour)
            if dist > farthest_dist:
                farthest_dist = dist
                farthest_city = city
        best_increase = float('inf')
        best_idx = 0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            increase = (distance_matrix[tour[i]][farthest_city] +
                        distance_matrix[farthest_city][tour[j]] -
                        distance_matrix[tour[i]][tour[j]])
            if increase < best_increase:
                best_increase = increase
                best_idx = j
        tour.insert(best_idx, farthest_city)
        unvisited.remove(farthest_city)
    best_tour = np.array(tour, dtype=int)
    report_best_tour(best_tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                delta = (distance_matrix[a][c] + distance_matrix[b][d] -
                         distance_matrix[a][b] - distance_matrix[c][d])
                if delta < 0:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
                    new_tour = np.array(tour, dtype=int)
                    if total_distance(distance_matrix, new_tour) < total_distance(distance_matrix, best_tour):
                        best_tour = new_tour
                        report_best_tour(best_tour)
    return best_tour