import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)

    best_tour = None
    best_cost = np.inf

    # Schedule: RCL sizes from n//2 down to sqrt(n)
    rcl_sizes = np.linspace(max(2, n//2), max(2, int(np.sqrt(n))), num=min(n, 10)).astype(int)
    rcl_sizes = np.unique(rcl_sizes)  # remove duplicates

    for rcl_size in rcl_sizes:
        # Single construction
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            dists = distance_matrix[current, list(unvisited)]
            sorted_indices = np.argsort(dists)
            sorted_cities = np.array(list(unvisited))[sorted_indices]
            k = min(rcl_size, len(sorted_cities))
            candidate = sorted_cities[np.random.randint(k)]
            tour.append(candidate)
            unvisited.remove(candidate)
            current = candidate
        tour = np.array(tour, dtype=np.int64)

        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == i + 1:
                        continue
                    a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
                    old = distance_matrix[a, b] + distance_matrix[c, d]
                    new = distance_matrix[a, c] + distance_matrix[b, d]
                    if new < old:
                        tour[i+1:j+1] = np.flip(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break

        # Compute cost
        cost = distance_matrix[tour[-1], tour[0]]
        for k in range(n-1):
            cost += distance_matrix[tour[k], tour[k+1]]

        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
            report_best_tour(best_tour)

    return best_tour