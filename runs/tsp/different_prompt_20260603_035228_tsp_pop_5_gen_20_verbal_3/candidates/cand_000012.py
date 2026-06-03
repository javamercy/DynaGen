import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n == 1:
        tour = np.array([0])
        report_best_tour(tour)
        return tour
    if n == 2:
        tour = np.array([0, 1])
        report_best_tour(tour)
        return tour

    # Nearest neighbor construction
    tour = [0]
    visited = [False] * n
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        best = -1
        bestd = float('inf')
        for j in range(n):
            if not visited[j] and distance_matrix[last, j] < bestd:
                bestd = distance_matrix[last, j]
                best = j
        tour.append(best)
        visited[best] = True

    def total_dist(t):
        idx = np.array(t)
        return distance_matrix[idx[-1], idx[0]] + np.sum(distance_matrix[idx[:-1], idx[1:]])

    best_tour = np.array(tour)
    best_dist = total_dist(best_tour)
    report_best_tour(best_tour)

    max_iters = 10
    for _ in range(max_iters):
        # 2-opt steepest descent
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_move = None
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[0]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[0]])
                    else:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[j+1]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[j+1]])
                    if delta > best_gain + 1e-12:
                        best_gain = delta
                        best_move = (i, j)
            if best_gain > 1e-12:
                i, j = best_move
                tour[i+1:j+1] = reversed(tour[i+1:j+1])
                improved = True
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)
        # Perturbation
        if _ < max_iters - 1:
            if np.random.random() < 0.5:
                # Double-bridge
                cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
                seg0 = tour[:cuts[0]]
                seg1 = tour[cuts[0]:cuts[1]]
                seg2 = tour[cuts[1]:cuts[2]]
                seg3 = tour[cuts[2]:]
                tour = seg0 + seg2 + seg1 + seg3
            else:
                # Random segment reversal
                i = np.random.randint(0, n - 1)
                j = np.random.randint(i + 2, n)
                tour[i+1:j+1] = reversed(tour[i+1:j+1])
    return best_tour