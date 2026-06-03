import numpy as np
def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n, dtype=int)
    best_tour = None
    best_dist = float('inf')
    for restart in range(10):
        # Construction: nearest neighbor
        tour = [0]
        unvisited = set(range(1, n))
        current = 0
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[current, x])
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        tour = np.array(tour, dtype=int)
        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    a = tour[i]
                    b = tour[(i + 1) % n]
                    c = tour[j]
                    d = tour[(j + 1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-9:
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        improved = True
        # Check if this tour is best so far
        dist = 0.0
        for k in range(n):
            dist += distance_matrix[tour[k], tour[(k + 1) % n]]
        if dist < best_dist - 1e-9:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        # Perturbation for next restart: double-bridge move on current tour
        if restart < 9:
            # Choose four random indices for the double-bridge
            idx = sorted(np.random.choice(range(1, n-1), 4, replace=False))
            a, b, c, d = idx
            # Double-bridge: reverse three segments
            tour = np.concatenate([
                tour[:a],
                tour[b:c],
                tour[a:b],
                tour[c:]
            ])
            # Ensure starting node remains 0? Not necessary, but ok.
    return best_tour