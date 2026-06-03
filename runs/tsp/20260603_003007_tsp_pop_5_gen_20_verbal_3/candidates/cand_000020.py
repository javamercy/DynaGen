import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0], dtype=np.int32)
        report_best_tour(tour)
        return tour
    
    def nn_tour(start):
        tour = np.empty(n, dtype=np.int32)
        unvisited = np.ones(n, dtype=bool)
        tour[0] = start
        unvisited[start] = False
        curr = start
        for i in range(1, n):
            dists = np.where(unvisited, distance_matrix[curr], np.inf)
            nxt = np.argmin(dists)
            tour[i] = nxt
            unvisited[nxt] = False
            curr = nxt
        return tour
    
    def tour_dist(tour):
        return distance_matrix[tour, np.roll(tour, -1)].sum()
    
    def two_opt(tour):
        ext = np.empty(n+1, dtype=np.int32)
        ext[:n] = tour
        ext[n] = tour[0]
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    delta = (distance_matrix[ext[i], ext[j]] +
                             distance_matrix[ext[i+1], ext[j+1]] -
                             distance_matrix[ext[i], ext[i+1]] -
                             distance_matrix[ext[j], ext[j+1]])
                    if delta < 0:
                        ext[i+1:j+1] = ext[i+1:j+1][::-1]
                        improved = True
                        new_tour = ext[:n].copy()
                        new_dist = tour_dist(new_tour)
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_tour = new_tour.copy()
                            report_best_tour(best_tour)
                        break
                if improved:
                    break
        return ext[:n], tour_dist(ext[:n])
    
    start_nodes = [0, n//4, n//2, 3*n//4]
    best_tour = None
    best_dist = np.inf
    for s in start_nodes:
        tour = nn_tour(s)
        dist = tour_dist(tour)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        tour, dist = two_opt(tour)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour