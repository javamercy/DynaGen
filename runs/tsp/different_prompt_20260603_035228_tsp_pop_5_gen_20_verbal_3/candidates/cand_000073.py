import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n, dtype=int)
        report_best_tour(tour)
        return tour

    def total_dist(t):
        idx = np.array(t, dtype=int)
        return distance_matrix[idx[-1], idx[0]] + np.sum(distance_matrix[idx[:-1], idx[1:]])

    def two_opt_first(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    if j == n-1:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[0]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[0]])
                    else:
                        delta = (distance_matrix[tour[i], tour[i+1]] +
                                 distance_matrix[tour[j], tour[j+1]] -
                                 distance_matrix[tour[i], tour[j]] -
                                 distance_matrix[tour[i+1], tour[j+1]])
                    if delta > 1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return tour

    def perturb(tour, method):
        if method == 'swap':
            i, j = np.random.choice(range(n), 2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]
        elif method == 'double_bridge':
            cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
            seg0 = tour[:cuts[0]]
            seg1 = tour[cuts[0]:cuts[1]]
            seg2 = tour[cuts[1]:cuts[2]]
            seg3 = tour[cuts[2]:]
            tour = np.concatenate([seg0, seg2, seg1, seg3])
        return tour

    best_tour = None
    best_dist = float('inf')
    num_restarts = 10
    max_iter = 200
    no_improve_limit = 20

    for _ in range(num_restarts):
        tour = np.random.permutation(n)
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        no_improve = 0
        for it in range(max_iter):
            tour = two_opt_first(tour)
            cur_dist = total_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= no_improve_limit:
                tour = np.random.permutation(n)
                no_improve = 0
            else:
                if no_improve < 5:
                    tour = perturb(tour, 'swap')
                else:
                    tour = perturb(tour, 'double_bridge')
    return best_tour