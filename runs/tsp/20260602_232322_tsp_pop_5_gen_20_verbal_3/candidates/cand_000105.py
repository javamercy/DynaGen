import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n, dtype=np.int64)
        report_best_tour(tour)
        return tour
    
    best_tour = None
    best_dist = np.inf
    
    for start in range(min(10, n)):
        unvisited = set(range(n))
        current = start
        tour = [current]
        unvisited.remove(current)
        while unvisited:
            nearest = min(unvisited, key=lambda city: distance_matrix[current, city])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        tour_arr = np.array(tour, dtype=np.int64)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour_arr[i], tour_arr[(i+1)%n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)
        
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        dist += delta
                        if dist < best_dist - 1e-10:
                            best_dist = dist
                            best_tour = np.array(tour, dtype=np.int64)
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
    return best_tour