import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = np.inf
    
    def tour_distance(tour):
        d = 0.0
        for k in range(n-1):
            d += distance_matrix[tour[k], tour[k+1]]
        d += distance_matrix[tour[-1], tour[0]]
        return d
    
    # Limited restarts from different start nodes
    starts = [0]
    if n >= 4:
        starts.extend([n//4, n//2, 3*n//4])
    else:
        starts = list(range(n))  # small n, try all
    
    for start in starts:
        # Nearest neighbor construction
        tour = np.empty(n, dtype=np.int32)
        unvisited = np.ones(n, dtype=bool)
        tour[0] = start
        unvisited[start] = False
        current = start
        for i in range(1, n):
            dists = np.where(unvisited, distance_matrix[current], np.inf)
            next_node = np.argmin(dists)
            tour[i] = next_node
            unvisited[next_node] = False
            current = next_node
        if np.any(tour == -1):
            continue  # just safety
        d = tour_distance(tour)
        if d < best_dist:
            best_dist = d
            best_tour = tour.copy()
            report_best_tour(best_tour)
        # 2-opt with full scan per pass, max passes = 2*n
        ext = np.empty(n+1, dtype=np.int32)
        ext[:n] = tour
        ext[n] = tour[0]
        max_passes = 2 * n
        for _ in range(max_passes):
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    delta = (distance_matrix[ext[i], ext[j]] +
                             distance_matrix[ext[i+1], ext[j+1]] -
                             distance_matrix[ext[i], ext[i+1]] -
                             distance_matrix[ext[j], ext[j+1]])
                    if delta < -1e-12:
                        ext[i+1:j+1] = ext[i+1:j+1][::-1]
                        improved = True
                        new_tour = ext[:n].copy()
                        d = tour_distance(new_tour)
                        if d < best_dist - 1e-12:
                            best_dist = d
                            best_tour = new_tour.copy()
                            report_best_tour(best_tour)
            if not improved:
                break
        # Also ensure best_tour reflects final tour of this start
        final_tour = ext[:n].copy()
        d = tour_distance(final_tour)
        if d < best_dist - 1e-12:
            best_dist = d
            best_tour = final_tour.copy()
            report_best_tour(best_tour)
    return best_tour