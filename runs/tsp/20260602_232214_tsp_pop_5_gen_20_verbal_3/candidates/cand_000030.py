import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # farthest pair initialization
    i, j = np.unravel_index(np.argmax(distance_matrix), (n, n))
    tour = [i, j]
    unvisited = set(range(n)) - {i, j}
    report_best_tour(np.array(tour))
    # farthest insertion construction
    while unvisited:
        best_city = None
        best_min_dist = -1.0
        for city in unvisited:
            min_dist = min(distance_matrix[city, t] for t in tour)
            if min_dist > best_min_dist + 1e-10:
                best_min_dist = min_dist
                best_city = city
        # insert best_city at cheapest position
        best_pos = None
        best_inc = float('inf')
        for k in range(len(tour)):
            a = tour[k]
            b = tour[(k+1) % len(tour)]
            cost = distance_matrix[a, best_city] + distance_matrix[best_city, b] - distance_matrix[a, b]
            if cost < best_inc - 1e-10:
                best_inc = cost
                best_pos = k+1
        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    # 2-opt improvement
    n_cities = n
    improved = True
    best_tour = tour_arr.copy()
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n_cities]] for i in range(n_cities))
    while improved:
        improved = False
        for i in range(n_cities-2):
            for j in range(i+2, n_cities):
                a = best_tour[i]
                b = best_tour[i+1]
                c = best_tour[j]
                d = best_tour[(j+1)%n_cities]
                delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                if delta < -1e-10:
                    new_tour = best_tour.copy()
                    new_tour[i+1:j+1] = best_tour[j:i:-1]
                    new_dist = best_dist + delta
                    best_tour = new_tour
                    best_dist = new_dist
                    improved = True
                    report_best_tour(best_tour)
    return best_tour