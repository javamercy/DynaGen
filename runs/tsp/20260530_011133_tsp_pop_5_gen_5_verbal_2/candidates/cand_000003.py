import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        return np.array([0])
    # farthest insertion construction
    # start with farthest pair
    start, end = np.unravel_index(np.argmax(distance_matrix), (n, n))
    tour = [start, end]
    unvisited = set(range(n)) - {start, end}
    while unvisited:
        # find farthest city from tour
        farthest_city = None
        farthest_dist = -1
        for city in unvisited:
            min_dist = min(distance_matrix[city][t] for t in tour)
            if min_dist > farthest_dist:
                farthest_dist = min_dist
                farthest_city = city
        # insert farthest_city at best position
        best_pos = 0
        best_increase = float('inf')
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            increase = distance_matrix[tour[i]][farthest_city] + distance_matrix[farthest_city][tour[j]] - distance_matrix[tour[i]][tour[j]]
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1
        tour.insert(best_pos, farthest_city)
        unvisited.remove(farthest_city)
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n-2):
            for k in range(i+2, n):
                if k == n-1 and i == 0:
                    continue  # skip wrap-around? Actually handle by reversing segment
                new_tour = tour_arr.copy()
                new_tour[i+1:k+1] = tour_arr[i+1:k+1][::-1]
                # compute delta
                a, b = tour_arr[i], tour_arr[i+1]
                c, d = tour_arr[k], tour_arr[(k+1)%n]
                delta = distance_matrix[a][new_tour[i+1]] + distance_matrix[new_tour[k]][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-10:
                    tour_arr = new_tour
                    improved = True
                    report_best_tour(tour_arr)
    return tour_arr