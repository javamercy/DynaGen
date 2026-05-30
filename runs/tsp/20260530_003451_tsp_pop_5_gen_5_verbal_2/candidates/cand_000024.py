import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # farthest insertion construction
    max_dist = -1
    start = 0
    end = 1
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i, j] > max_dist:
                max_dist = distance_matrix[i, j]
                start, end = i, j
    tour_list = [start, end]
    in_tour = {start, end}
    while len(tour_list) < n:
        farthest_city = None
        max_min_dist = -1
        for city in range(n):
            if city in in_tour:
                continue
            min_dist = min(distance_matrix[city, t] for t in tour_list)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                farthest_city = city
        best_pos = 0
        best_increase = float('inf')
        for pos in range(len(tour_list)):
            prev = tour_list[pos]
            nxt = tour_list[(pos+1) % len(tour_list)]
            increase = distance_matrix[prev, farthest_city] + distance_matrix[farthest_city, nxt] - distance_matrix[prev, nxt]
            if increase < best_increase:
                best_increase = increase
                best_pos = pos+1
        tour_list.insert(best_pos, farthest_city)
        in_tour.add(farthest_city)
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
        # double-bridge perturbation
        i = np.random.randint(0, n//4)
        j = i + 1 + np.random.randint(0, n//4)
        k = j + 1 + np.random.randint(0, n//4)
        l = k + 1 + np.random.randint(0, n - k)
        if l >= n:
            l = n - 1
        # ensure distinct segments
        if i >= j or j >= k or k >= l:
            continue
        tour = best_tour.copy().tolist()
        # double-bridge: [0:i] + [j:k] + [i:j] + [k:l] + [l:]
        new_tour = tour[:i] + tour[j:k] + tour[i:j] + tour[k:l] + tour[l:]
        # convert back to list for in-place modification
        tour = new_tour
        two_opt(tour)
        cur_dist = sum(distance_matrix[tour[k], tour[(k+1)%n]] for k in range(n))
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)
    return best_tour