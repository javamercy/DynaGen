import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    # nearest neighbor heuristic
    def nearest_neighbor():
        start = 0
        tour = [start]
        visited = {start}
        current = start
        for _ in range(n-1):
            best = None
            best_dist = np.inf
            for v in range(n):
                if v not in visited and distance_matrix[current, v] < best_dist:
                    best_dist = distance_matrix[current, v]
                    best = v
            tour.append(best)
            visited.add(best)
            current = best
        return np.array(tour)

    def tour_cost(tour):
        total = 0.0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i+1)%n]]
        return total

    tour = nearest_neighbor()
    current_cost = tour_cost(tour)
    best_tour = tour.copy()
    best_cost = current_cost
    report_best_tour(tour)

    # if all distances zero (unlikely), set T0 to 1
    T0 = current_cost * 0.5 if current_cost > 0 else 1.0
    max_iter = 5000 * n
    rng = np.random.default_rng()

    for it in range(max_iter):
        # linear cooling
        fraction = (it + 1) / max_iter
        T = T0 * (1 - fraction)
        if T == 0:
            T = 1e-12  # avoid division by zero

        # 2-opt move
        i = rng.integers(0, n-2)
        j = rng.integers(i+2, n)
        a = tour[i]
        b = tour[i+1]
        c = tour[j]
        d = tour[(j+1)%n]
        delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # reverse segment [i+1, j]
            tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = tour.copy()
                report_best_tour(best_tour)

    return best_tour