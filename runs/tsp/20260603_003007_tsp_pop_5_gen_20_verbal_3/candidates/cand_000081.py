import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour

    def tour_distance(tour):
        d = 0.0
        for k in range(n - 1):
            d += distance_matrix[tour[k], tour[k+1]]
        d += distance_matrix[tour[-1], tour[0]]
        return d

    best_tour = None
    best_dist = np.inf

    starts = [0]
    if n >= 4:
        starts.extend([n//4, n//2, 3*n//4])
    else:
        starts = list(range(n))

    for start in starts:
        # Nearest neighbor construction
        tour = np.empty(n, dtype=np.int32)
        unvisited = np.ones(n, dtype=bool)
        tour[0] = start
        unvisited[start] = False
        current = start
        for i in range(1, n):
            dists = np.where(unvisited, distance_matrix[current], np.inf)
            nxt = np.argmin(dists)
            tour[i] = nxt
            unvisited[nxt] = False
            current = nxt
        d = tour_distance(tour)
        if d < best_dist - 1e-12:
            best_dist = d
            best_tour = tour.copy()
            report_best_tour(best_tour)

        # 2-opt improvement
        ext = np.empty(n + 1, dtype=np.int32)
        ext[:n] = tour
        ext[n] = tour[0]
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    delta = (distance_matrix[ext[i], ext[j]] +
                             distance_matrix[ext[i+1], ext[j+1]] -
                             distance_matrix[ext[i], ext[i+1]] -
                             distance_matrix[ext[j], ext[j+1]])
                    if delta < -1e-12:
                        ext[i+1:j+1] = ext[i+1:j+1][::-1]
                        improved = True
                        # evaluate new tour
                        new_tour = ext[:n].copy()
                        d_new = tour_distance(new_tour)
                        if d_new < best_dist - 1e-12:
                            best_dist = d_new
                            best_tour = new_tour.copy()
                            report_best_tour(best_tour)
        # final tour from this start
        final_tour = ext[:n].copy()
        d_final = tour_distance(final_tour)
        if d_final < best_dist - 1e-12:
            best_dist = d_final
            best_tour = final_tour.copy()
            report_best_tour(best_tour)

    return best_tour