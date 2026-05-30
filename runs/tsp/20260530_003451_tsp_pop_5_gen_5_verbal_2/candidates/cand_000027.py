import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # nearest neighbor construction
    start = np.random.randint(n)
    tour_list = [start]
    in_tour = {start}
    cur = start
    while len(tour_list) < n:
        next_city = None
        min_dist = np.inf
        for city in range(n):
            if city not in in_tour and distance_matrix[cur, city] < min_dist:
                min_dist = distance_matrix[cur, city]
                next_city = city
        tour_list.append(next_city)
        in_tour.add(next_city)
        cur = next_city
    best_tour = np.array(tour_list)
    best_dist = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    report_best_tour(best_tour)
    # exhaustive 2-opt
    def two_opt(tour):
        nonlocal best_tour, best_dist
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                    if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        cur_dist = sum(distance_matrix[tour[k], tour[(k+1)%n]] for k in range(n))
                        if cur_dist < best_dist:
                            best_dist = cur_dist
                            best_tour = np.array(tour)
                            report_best_tour(best_tour)
        return tour
    tour = tour_list.copy()
    two_opt(tour)
    # iterated local search with double-bridge kick
    for _ in range(10):
        i = np.random.randint(0, n//4)
        j = i + 1 + np.random.randint(0, n//4)
        k = j + 1 + np.random.randint(0, n//4)
        l = k + 1 + np.random.randint(0, n - k)
        if l >= n:
            l = n - 1
        if i >= j or j >= k or k >= l:
            continue
        tour = best_tour.copy().tolist()
        new_tour = tour[:i] + tour[j:k] + tour[i:j] + tour[k:l] + tour[l:]
        tour = new_tour
        two_opt(tour)
        cur_dist = sum(distance_matrix[tour[k], tour[(k+1)%n]] for k in range(n))
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)
    return best_tour