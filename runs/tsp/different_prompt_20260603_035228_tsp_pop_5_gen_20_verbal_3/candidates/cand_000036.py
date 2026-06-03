import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n, dtype=int)
    
    def tour_distance(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    
    def nearest_neighbor_from(start):
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            nxt = min(unvisited, key=lambda x: distance_matrix[current, x])
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return np.array(tour, dtype=int)
    
    def two_opt_steepest(tour):
        improved = True
        while improved:
            improved = False
            best_delta = 0.0
            best_i = best_j = -1
            for i in range(n-1):
                for j in range(i+2, n):
                    a, b = tour[i], tour[(i+1)%n]
                    c, d = tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_i, best_j = i, j
                        improved = True
            if improved:
                tour[best_i+1:best_j+1] = tour[best_i+1:best_j+1][::-1]
        return tour
    
    def double_bridge(tour):
        if n < 4:
            return tour
        indices = sorted(np.random.choice(n, 4, replace=False))
        a, b, c, d = indices
        return np.concatenate([tour[:a], tour[c:d], tour[b:c], tour[a:b], tour[d:]])
    
    best_tour = None
    best_dist = np.inf
    
    for run in range(5):
        start = np.random.randint(n)
        tour = nearest_neighbor_from(start)
        tour = two_opt_steepest(tour)
        dist = tour_distance(tour)
        if dist < best_dist - 1e-12:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        
        for cycle in range(10):
            tour = double_bridge(tour)
            tour = two_opt_steepest(tour)
            dist = tour_distance(tour)
            if dist < best_dist - 1e-12:
                best_dist = dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
    
    return best_tour