import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    # Cheapest insertion construction
    start = 0
    d = distance_matrix[start].copy()
    d[start] = np.inf
    nearest = np.argmin(d)
    tour = [start, nearest]
    unvisited = set(range(n)) - {start, nearest}
    
    while unvisited:
        best_cost = np.inf
        best_city = None
        best_pos = None
        for city in unvisited:
            for pos in range(len(tour)):
                prev = tour[pos]
                next = tour[(pos + 1) % len(tour)]
                cost = distance_matrix[prev, city] + distance_matrix[city, next] - distance_matrix[prev, next]
                if cost < best_cost:
                    best_cost = cost
                    best_city = city
                    best_pos = pos + 1
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Reverse segment from i+1 to j
                new_tour = tour_arr.copy()
                new_tour[i + 1:j + 1] = tour_arr[i + 1:j + 1][::-1]
                # Compute delta
                a = tour_arr[i]
                b = tour_arr[i + 1]
                c = tour_arr[j]
                d = tour_arr[(j + 1) % n]
                delta = distance_matrix[a][new_tour[i + 1]] + distance_matrix[new_tour[j]][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-12:
                    tour_arr = new_tour
                    improved = True
                    report_best_tour(tour_arr)
    return tour_arr