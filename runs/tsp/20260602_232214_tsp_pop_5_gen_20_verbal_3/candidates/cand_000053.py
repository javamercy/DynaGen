import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        return np.arange(n)
    
    def tour_dist(tour):
        total = 0.0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i+1)%n]]
        return total
    
    def two_opt(tour):
        best = tour.copy()
        best_dist = tour_dist(best)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a, b = best[i], best[(i+1)%n]
                    c, d = best[j], best[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-12:
                        new_tour = np.concatenate([best[:i+1], best[i+1:j+1][::-1], best[j+1:]])
                        best_dist += delta
                        best = new_tour
                        improved = True
        return best, best_dist
    
    def nearest_neighbor(start):
        tour = [start]
        unvisited = set(range(n)) - {start}
        while unvisited:
            last = tour[-1]
            next_city = min(unvisited, key=lambda x: distance_matrix[last, x])
            tour.append(next_city)
            unvisited.remove(next_city)
        return np.array(tour)
    
    best_tour = None
    best_dist = float('inf')
    starts = set([0])
    if n > 1:
        extra = np.random.choice(range(1, n), size=min(2, n-1), replace=False).tolist()
        starts.update(extra)
    for start in starts:
        tour = nearest_neighbor(start)
        tour, dist = two_opt(tour)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour
            report_best_tour(best_tour)
    return best_tour