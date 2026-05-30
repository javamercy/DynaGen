import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    
    # Cheapest insertion construction
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        best_city = None
        best_pos = None
        best_inc = float('inf')
        for city in unvisited:
            # compute insertion cost at each position
            for pos in range(len(tour) + 1):
                if pos == 0:
                    i = tour[-1]
                    j = tour[0]
                elif pos == len(tour):
                    i = tour[-1]
                    j = tour[0]
                else:
                    i = tour[pos-1]
                    j = tour[pos]
                inc = distance_matrix[i, city] + distance_matrix[city, j] - distance_matrix[i, j]
                if inc < best_inc:
                    best_inc = inc
                    best_city = city
                    best_pos = pos
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    
    best_tour = np.array(tour, dtype=int)
    report_best_tour(best_tour)
    
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                a = tour[i]
                b = tour[(i+1) % n]
                c = tour[j]
                d = tour[(j+1) % n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < 0:
                    # reverse segment i+1 .. j
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
                    new_tour = np.array(tour, dtype=int)
                    # compute new distance
                    new_dist = 0.0
                    for k in range(n):
                        new_dist += distance_matrix[new_tour[k], new_tour[(k+1)%n]]
                    if new_dist < total_distance(distance_matrix, best_tour):
                        best_tour = new_tour
                        report_best_tour(best_tour)
    return best_tour

def total_distance(dist, tour):
    n = len(tour)
    return sum(dist[tour[i], tour[(i+1)%n]] for i in range(n))