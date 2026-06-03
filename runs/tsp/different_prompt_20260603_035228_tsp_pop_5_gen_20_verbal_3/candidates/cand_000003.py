import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 1:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # Nearest neighbor construction
    tour = [0]
    visited = [False] * n
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        best = -1
        bestd = float('inf')
        for j in range(n):
            if not visited[j] and distance_matrix[last, j] < bestd:
                bestd = distance_matrix[last, j]
                best = j
        tour.append(best)
        visited[best] = True
    # Total distance
    total = sum(distance_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))
    best_total = total
    best_tour = tour[:]
    report_best_tour(np.array(tour))
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                # delta = old - new
                if j == n - 1:
                    delta = (distance_matrix[tour[i], tour[i+1]] +
                             distance_matrix[tour[j], tour[0]] -
                             distance_matrix[tour[i], tour[j]] -
                             distance_matrix[tour[i+1], tour[0]])
                else:
                    delta = (distance_matrix[tour[i], tour[i+1]] +
                             distance_matrix[tour[j], tour[j+1]] -
                             distance_matrix[tour[i], tour[j]] -
                             distance_matrix[tour[i+1], tour[j+1]])
                if delta > 0:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
                    total -= delta
                    if total < best_total:
                        best_total = total
                        best_tour = tour[:]
                        report_best_tour(np.array(tour))
    # Return best found
    tour_arr = np.array(best_tour)
    report_best_tour(tour_arr)
    return tour_arr