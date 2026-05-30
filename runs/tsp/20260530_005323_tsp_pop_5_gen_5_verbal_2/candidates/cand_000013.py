import numpy as np

def total_distance(dist, tour):
    return sum(dist[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    # farthest insertion construction
    start, end = np.unravel_index(np.argmax(distance_matrix), (n, n))
    tour = [start, end]
    unvisited = set(range(n)) - {start, end}
    while unvisited:
        farthest_dist = -1
        farthest_city = -1
        for city in unvisited:
            dist = min(distance_matrix[city][t] for t in tour)
            if dist > farthest_dist:
                farthest_dist = dist
                farthest_city = city
        best_increase = float('inf')
        best_idx = 0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            increase = (distance_matrix[tour[i]][farthest_city] +
                        distance_matrix[farthest_city][tour[j]] -
                        distance_matrix[tour[i]][tour[j]])
            if increase < best_increase:
                best_increase = increase
                best_idx = j
        tour.insert(best_idx, farthest_city)
        unvisited.remove(farthest_city)
    best_tour = np.array(tour, dtype=int)
    report_best_tour(best_tour)
    
    def two_opt(tour_list):
        improved = True
        best_list = tour_list[:]
        best_dist = total_distance(distance_matrix, best_list)
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue
                    a, b = best_list[i], best_list[(i+1)%n]
                    c, d = best_list[j], best_list[(j+1)%n]
                    delta = (distance_matrix[a][c] + distance_matrix[b][d] -
                             distance_matrix[a][b] - distance_matrix[c][d])
                    if delta < 0:
                        best_list[i+1:j+1] = reversed(best_list[i+1:j+1])
                        improved = True
                        new_dist = total_distance(distance_matrix, best_list)
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_tour_arr = np.array(best_list, dtype=int)
                            report_best_tour(best_tour_arr)
        return best_list, best_dist
    
    # initial 2-opt
    tour, best_dist = two_opt(tour)
    # double-bridge perturbations and restarts
    for _ in range(5):
        # make a copy
        current = tour[:]
        # double bridge: pick 3 distinct split points (exclude endpoints)
        if n < 4:
            break
        # ensure segments are non-empty: choose i, j, k such that i < j < k and i>=1, k<=n-1
        i = np.random.randint(1, n // 2)
        j = np.random.randint(i+1, n - 1)
        k = np.random.randint(j+1, n)
        # segments: A=[0:i], B=[i:j], C=[j:k], D=[k:]
        A = current[:i]
        B = current[i:j]
        C = current[j:k]
        D = current[k:]
        # reorder as A + C + B + D (or other pattern)
        new_tour = A + C + B + D
        # run 2-opt on new tour
        new_tour, new_dist = two_opt(new_tour)
        if new_dist < best_dist:
            best_dist = new_dist
            tour = new_tour
            best_tour = np.array(tour, dtype=int)
            report_best_tour(best_tour)
    return np.array(tour, dtype=int)