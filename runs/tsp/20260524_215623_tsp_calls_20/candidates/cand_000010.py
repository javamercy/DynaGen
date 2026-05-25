import numpy as np
def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # farthest pair
    max_dist = -1
    start = 0
    second = 1
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i, j]
            if d > max_dist:
                max_dist = d
                start, second = i, j
    tour = [start, second]
    visited = {start, second}
    used = 0
    # regret insertion with tie-breaking by farthest
    while used < budget and len(tour) < n:
        best_regret = -1.0
        best_city = -1
        best_pos = -1
        best_inc = None
        for city in range(n):
            if city in visited:
                continue
            m = len(tour)
            incs = np.zeros(m)
            for i in range(m):
                a = tour[i]
                b = tour[(i+1)%m]
                incs[i] = distance_matrix[a,city] + distance_matrix[city,b] - distance_matrix[a,b]
            sorted_incs = np.sort(incs)
            regret = sorted_incs[1] - sorted_incs[0]
            min_inc = incs[np.argmin(incs)]
            if regret > best_regret or (abs(regret - best_regret) < 1e-12 and (best_inc is None or min_inc > best_inc)):
                best_regret = regret
                best_city = city
                best_pos = int(np.argmin(incs))
                best_inc = min_inc
        tour.insert(best_pos+1, best_city)
        visited.add(best_city)
        used += 1
    # ensure complete tour
    if len(tour) < n:
        remaining = list(set(range(n)) - visited)
        np.random.shuffle(remaining)
        tour = tour + remaining
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    # 2-opt with restart (reset i after each improvement)
    improved = True
    while used < budget and improved:
        improved = False
        i = 0
        while i < n - 1 and used < budget:
            j = i + 2
            while j < n and used < budget:
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                delta = -distance_matrix[a,b] - distance_matrix[c,d] + distance_matrix[a,c] + distance_matrix[b,d]
                if delta < -1e-12:
                    new_tour = tour.copy()
                    new_tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    tour = new_tour
                    used += 1
                    report_best_tour(np.array(tour))
                    improved = True
                    break
                j += 1
            if improved:
                break
            i += 1
    return np.array(tour)