import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour
    if n == 2:
        tour = np.array([0, 1])
        report_best_tour(tour)
        return tour

    def total_dist(t):
        idx = np.array(t)
        return distance_matrix[idx[-1], idx[0]] + np.sum(distance_matrix[idx[:-1], idx[1:]])

    def two_opt_best(tour):
        improved = True
        while improved:
            improved = False
            best_i = -1
            best_j = -1
            best_delta = 0.0
            for i in range(n-1):
                for j in range(i+2, n):
                    if j == n-1:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[0]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[0]])
                    else:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[j+1]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[j+1]])
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_delta < -1e-12:
                i, j = best_i, best_j
                if j == n-1:
                    tour[i+1:] = reversed(tour[i+1:])
                else:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                improved = True
        return tour

    def nearest_neighbor(start):
        tour = [start]
        visited = {start}
        for _ in range(n-1):
            last = tour[-1]
            min_dist = np.inf
            next_node = None
            for v in range(n):
                if v not in visited:
                    d = distance_matrix[last, v]
                    if d < min_dist:
                        min_dist = d
                        next_node = v
            tour.append(next_node)
            visited.add(next_node)
        return tour

    best_tour = None
    best_dist = float('inf')
    restarts = 10
    for _ in range(restarts):
        start_node = np.random.randint(n)
        tour = nearest_neighbor(start_node)
        tour = two_opt_best(tour)
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)
    return best_tour