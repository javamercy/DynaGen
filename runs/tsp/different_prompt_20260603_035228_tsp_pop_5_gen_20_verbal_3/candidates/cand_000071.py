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

    def two_opt_steepest(tour):
        improved = True
        while improved:
            improved = False
            best_gain = 0.0
            best_i = best_j = None
            for i in range(n - 2):
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
                        best_i, best_j = i, j
            if best_gain > 1e-12:
                i, j = best_i, best_j
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
                improved = True
        return tour

    def double_bridge(tour):
        cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
        seg0 = tour[:cuts[0]]
        seg1 = tour[cuts[0]:cuts[1]]
        seg2 = tour[cuts[1]:cuts[2]]
        seg3 = tour[cuts[2]:]
        return np.concatenate([seg0, seg2, seg1, seg3])

    def nearest_insertion():
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        # build tour with nearest insertion
        for _ in range(n - 1):
            # find unvisited city nearest to current tour
            best_city = None
            best_dist = float('inf')
            best_idx = 0
            for u in unvisited:
                for idx in range(len(tour)):
                    d = distance_matrix[tour[idx], u]
                    if d < best_dist:
                        best_dist = d
                        best_city = u
                        best_idx = idx
            # insert best_city at best position (minimizing increase in tour length)
            # actually we just insert next to the closest tour city; we can improve insertion position
            # for simplicity, insert after best_idx
            tour.insert(best_idx + 1, best_city)
            unvisited.remove(best_city)
        return np.array(tour, dtype=int)

    best_tour = None
    best_dist = float('inf')
    num_restarts = 15
    max_cycles = 40
    stall_limit = 8

    for _ in range(num_restarts):
        tour = nearest_insertion()
        tour = two_opt_steepest(tour)
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        no_improve = 0
        for cycle in range(max_cycles):
            tour = two_opt_steepest(tour)
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
                tour = nearest_insertion()
                tour = two_opt_steepest(tour)
                no_improve = 0
            else:
                if no_improve == 0:
                    seg_len = np.random.randint(2, max(4, n // 3 + 1))
                    i = np.random.randint(0, n - seg_len)
                    tour = tour.copy()
                    tour[i:i+seg_len] = tour[i:i+seg_len][::-1]
                else:
                    tour = double_bridge(tour)

    return best_tour