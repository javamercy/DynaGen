import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    
    def tour_length(tour):
        return distance_matrix[tour, np.roll(tour, -1)].sum()
    
    best_tour = None
    best_dist = float('inf')
    R = max(1, min(5, n))
    for start in range(R):
        # nearest neighbor
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
        
        dist = tour_length(tour)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        
        # multi-pass 2-opt
        ext = np.empty(n+1, dtype=np.int32)
        ext[:n] = tour
        ext[n] = tour[0]
        for _ in range(10):  # max 10 passes
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    delta = (distance_matrix[ext[i], ext[j]] +
                             distance_matrix[ext[i+1], ext[j+1]] -
                             distance_matrix[ext[i], ext[i+1]] -
                             distance_matrix[ext[j], ext[j+1]])
                    if delta < -1e-10:
                        ext[i+1:j+1] = ext[i+1:j+1][::-1]
                        improved = True
            if not improved:
                break
            new_tour = ext[:n].copy()
            new_dist = tour_length(new_tour)
            if new_dist < best_dist:
                best_dist = new_dist
                best_tour = new_tour.copy()
                report_best_tour(best_tour)
    return best_tour