import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    # Cheapest insertion initialization
    start = 0
    d = distance_matrix[start].copy()
    d[start] = np.inf
    nearest = np.argmin(d)
    tour = [start, nearest]
    unvisited = set(range(n)) - {start, nearest}
    
    while unvisited:
        best_inc = np.inf
        best_city = None
        best_pos = None
        for city in unvisited:
            for pos in range(len(tour)):
                prev = tour[pos]
                nxt = tour[(pos+1) % len(tour)]
                cost = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                if cost < best_inc:
                    best_inc = cost
                    best_city = city
                    best_pos = pos + 1
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    
    best_tour = np.array(tour)
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                new_tour = np.concatenate([best_tour[:i+1], best_tour[j:i:-1], best_tour[j+1:]])
                new_dist = sum(distance_matrix[new_tour[k], new_tour[(k+1)%n]] for k in range(n))
                if new_dist < best_dist - 1e-12:
                    best_dist = new_dist
                    best_tour = new_tour
                    improved = True
                    report_best_tour(best_tour)
    return best_tour