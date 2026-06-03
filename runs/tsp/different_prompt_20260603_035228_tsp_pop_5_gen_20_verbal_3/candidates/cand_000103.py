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

    def two_opt_first_improve(tour):
        improved = True
        while improved:
            improved = False
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
                    if delta > 1e-12:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return tour

    def double_bridge(tour):
        cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
        seg0 = tour[:cuts[0]]
        seg1 = tour[cuts[0]:cuts[1]]
        seg2 = tour[cuts[1]:cuts[2]]
        seg3 = tour[cuts[2]:]
        return np.concatenate([seg0, seg2, seg1, seg3])

    def stochastic_nn(start):
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            last = tour[-1]
            dists = np.array([distance_matrix[last, j] for j in unvisited])
            inv_dists = 1.0 / (dists + 1e-12)
            probs = inv_dists / inv_dists.sum()
            chosen = np.random.choice(list(unvisited), p=probs)
            tour.append(chosen)
            unvisited.remove(chosen)
        return np.array(tour, dtype=int)

    best_tour = None
    best_dist = float('inf')
    num_restarts = 10
    max_cycles = 30
    stall_limit = 8

    for _ in range(num_restarts):
        start = np.random.randint(n)
        tour = stochastic_nn(start)
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        no_improve = 0
        for cycle in range(max_cycles):
            tour = two_opt_first_improve(tour)
            cur_dist = total_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1

            if cycle == max_cycles - 1:
                break

            if no_improve >= stall_limit:
                start = np.random.randint(n)
                tour = stochastic_nn(start)
                no_improve = 0
            else:
                tour = double_bridge(tour)

    return best_tour