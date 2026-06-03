import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = float('inf')
    num_restarts = max(1, min(10, n // 20))
    for _ in range(num_restarts):
        tour = np.random.permutation(n).astype(np.int32)
        ext = np.empty(n + 1, dtype=np.int32)
        ext[:n] = tour
        ext[n] = tour[0]
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 2, n):
                    delta = (distance_matrix[ext[i], ext[j]] +
                             distance_matrix[ext[i + 1], ext[j + 1]] -
                             distance_matrix[ext[i], ext[i + 1]] -
                             distance_matrix[ext[j], ext[j + 1]])
                    if delta < 0:
                        ext[i + 1:j + 1] = ext[i + 1:j + 1][::-1]
                        improved = True
            if improved:
                new_tour = ext[:n].copy()
                new_dist = distance_matrix[new_tour, np.roll(new_tour, -1)].sum()
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_tour = new_tour.copy()
                    report_best_tour(best_tour)
        if not improved:  # local minimum reached, still check against best
            final_tour = ext[:n].copy()
            final_dist = distance_matrix[final_tour, np.roll(final_tour, -1)].sum()
            if final_dist < best_dist:
                best_dist = final_dist
                best_tour = final_tour.copy()
                report_best_tour(best_tour)
    return best_tour