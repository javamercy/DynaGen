import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0, 1])

    rng = np.random.default_rng()
    best_tour = None
    best_len = float('inf')

    # Quick nearest neighbor for initial valid tour
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda x: distance_matrix[last, x])
        tour.append(next_city)
        unvisited.remove(next_city)
    tour = np.array(tour, dtype=int)
    length = distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])
    best_tour = tour.copy()
    best_len = length
    report_best_tour(best_tour)

    max_restarts = 10
    for _ in range(max_restarts):
        # Randomized greedy construction with RCL
        start = rng.integers(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        rcl_size = max(1, int(np.sqrt(n)))
        while unvisited:
            current = tour[-1]
            dists = [(city, distance_matrix[current, city]) for city in unvisited]
            dists.sort(key=lambda x: x[1])
            rcl = dists[:rcl_size]
            next_city = rng.choice([x[0] for x in rcl]) if rcl else list(unvisited)[0]
            tour.append(next_city)
            unvisited.remove(next_city)
        tour = np.array(tour, dtype=int)
        length = distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])

        # 2-opt improvement
        tour = tour.tolist()
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for k in range(i+2, n):
                    a = tour[i]
                    b = tour[i+1]
                    c = tour[k]
                    d = tour[(k+1) % n]
                    old = distance_matrix[a, b] + distance_matrix[c, d]
                    new = distance_matrix[a, c] + distance_matrix[b, d]
                    if new + 1e-10 < old:
                        tour[i+1:k+1] = tour[i+1:k+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        tour = np.array(tour, dtype=int)
        length = distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])
        if length < best_len - 1e-10:
            best_len = length
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour