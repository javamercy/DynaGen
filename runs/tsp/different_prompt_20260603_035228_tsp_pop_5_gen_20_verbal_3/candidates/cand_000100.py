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

    def steepest_two_opt(tour):
        improved = True
        while improved:
            improved = False
            best_gain = 0.0
            best_i = best_j = -1
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

    def farthest_insertion():
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            # find farthest unvisited city
            max_min_dist = -1
            far_city = -1
            for c in unvisited:
                min_dist = min(distance_matrix[c, t] for t in tour)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    far_city = c
            # insert far_city at best position
            best_cost = float('inf')
            best_pos = 0
            L = len(tour)
            for i in range(L):
                j = (i + 1) % L
                cost_inc = (distance_matrix[tour[i], far_city] +
                            distance_matrix[far_city, tour[j]] -
                            distance_matrix[tour[i], tour[j]])
                if cost_inc < best_cost:
                    best_cost = cost_inc
                    best_pos = i + 1
            tour.insert(best_pos, far_city)
            unvisited.remove(far_city)
        return np.array(tour, dtype=int)

    best_tour = None
    best_dist = float('inf')
    num_restarts = 10
    max_cycles = 30
    stall_limit = 8

    for _ in range(num_restarts):
        tour = farthest_insertion()
        tour = steepest_two_opt(tour)
        cur_dist = total_dist(tour)
        if cur_dist < best_dist - 1e-12:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)

        no_improve = 0
        for cycle in range(max_cycles):
            if no_improve < 5:
                seg_len = np.random.randint(2, max(4, n//3 + 1))
                i = np.random.randint(0, n - seg_len)
                tour[i:i+seg_len] = tour[i:i+seg_len][::-1]
            else:
                tour = double_bridge(tour)
            tour = steepest_two_opt(tour)
            cur_dist = total_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= stall_limit:
                tour = farthest_insertion()
                tour = steepest_two_opt(tour)
                cur_dist = total_dist(tour)
                if cur_dist < best_dist - 1e-12:
                    best_dist = cur_dist
                    best_tour = tour.copy()
                    report_best_tour(best_tour)
                no_improve = 0

    return best_tour