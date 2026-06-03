import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    best_tour = None
    best_dist = np.inf
    num_restarts = min(20, max(10, n // 10))
    for _ in range(num_restarts):
        start = np.random.randint(n)
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            next_city = min(unvisited, key=lambda x: distance_matrix[cur, x])
            tour.append(next_city)
            unvisited.remove(next_city)
            cur = next_city
        tour = np.array(tour, dtype=int)
        dist = sum(distance_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))
        improved = True
        while improved:
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n):
                    if j - i == 1:
                        continue
                    a, b = tour[i], tour[(i + 1) % n]
                    c, d = tour[j], tour[(j + 1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-12:
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        improved = True
                        dist += delta
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour