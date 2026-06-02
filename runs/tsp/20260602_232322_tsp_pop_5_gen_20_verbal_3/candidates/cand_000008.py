import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # start with triangle
    tour = [0, 1, 2]
    remaining = set(range(3, n))
    # insertion cost function
    def delta(city, pos):
        before = tour[pos-1]
        after = tour[pos] if pos < len(tour) else tour[0]
        return distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
    # insert all remaining using cheapest insertion
    while remaining:
        best_city = -1
        best_pos = -1
        best_cost = float('inf')
        for city in remaining:
            for pos in range(len(tour)):
                cost = delta(city, pos)
                if cost < best_cost - 1e-10:
                    best_cost = cost
                    best_city = city
                    best_pos = pos
        # insert best city at best position
        tour.insert(best_pos, best_city)
        remaining.remove(best_city)
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                if j - i < n:
                    new_tour = tour.copy()
                    new_tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    old = distance_matrix[a,b] + distance_matrix[c,d]
                    new = distance_matrix[a,c] + distance_matrix[b,d]
                    if new < old - 1e-10:
                        tour = new_tour
                        tour_arr = np.array(tour)
                        report_best_tour(tour_arr)
                        improved = True
                        break
            if improved:
                break
    return np.array(tour)