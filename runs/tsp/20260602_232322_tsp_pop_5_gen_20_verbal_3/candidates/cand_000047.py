import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)
    if n == 2:
        tour = np.array([0, 1], dtype=np.int64)
        report_best_tour(tour.copy())
        return tour

    # Nearest neighbor from random start
    start = np.random.randint(n)
    tour = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    tour[0] = start
    used[start] = True
    for i in range(1, n):
        last = tour[i-1]
        best_dist = np.inf
        best_city = -1
        for j in range(n):
            if not used[j]:
                d = distance_matrix[last, j]
                if d < best_dist:
                    best_dist = d
                    best_city = j
        tour[i] = best_city
        used[best_city] = True

    # Compute cost
    cost = distance_matrix[tour[-1], tour[0]]
    for k in range(n-1):
        cost += distance_matrix[tour[k], tour[k+1]]
    best_tour = tour.copy()
    best_cost = cost
    report_best_tour(best_tour.copy())

    # SA parameters
    T = cost / 10.0
    cooling_rate = 0.999
    max_iter = max(10000, n * 100)
    for _ in range(max_iter):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i > j:
            i, j = j, i
        if j - i < 2:
            continue
        a = tour[i]
        b = tour[(i+1) % n]
        c = tour[j]
        d = tour[(j+1) % n]
        old = distance_matrix[a, b] + distance_matrix[c, d]
        new = distance_matrix[a, c] + distance_matrix[b, d]
        delta = new - old
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            tour[i+1:j+1] = np.flip(tour[i+1:j+1])
            cost += delta
            if cost < best_cost:
                best_cost = cost
                best_tour = tour.copy()
                report_best_tour(best_tour.copy())
        T *= cooling_rate
        if T < 1e-10:
            break
    return best_tour