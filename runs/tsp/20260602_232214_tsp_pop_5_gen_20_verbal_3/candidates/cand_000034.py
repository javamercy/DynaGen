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
    best_tour = np.array(tour)
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)

    # 2-opt improvement
    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = tour[i]; b = tour[i+1]; c = tour[j]; d = tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        tour[i+1:j+1] = tour[j:i:-1]
                        improved = True
        return tour

    # double-bridge perturbation
    def double_bridge(tour):
        a = np.random.randint(1, n//3)
        b = np.random.randint(a+1, 2*n//3)
        c = np.random.randint(b+1, n-1)
        seg1 = tour[:a]
        seg2 = tour[a:b]
        seg3 = tour[b:c]
        seg4 = tour[c:]
        new_tour = np.concatenate([seg1, seg3, seg2, seg4])
        return new_tour

    current_tour = best_tour.copy()
    for _ in range(10):
        current_tour = two_opt(current_tour)
        current_dist = sum(distance_matrix[current_tour[i], current_tour[(i+1)%n]] for i in range(n))
        if current_dist < best_dist - 1e-10:
            best_dist = current_dist
            best_tour = current_tour.copy()
            report_best_tour(best_tour)
        current_tour = double_bridge(best_tour.copy())
    return best_tour