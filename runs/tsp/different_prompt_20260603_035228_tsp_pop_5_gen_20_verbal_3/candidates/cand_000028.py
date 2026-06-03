import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def total_dist(tour):
        idx = np.array(tour)
        return distance_matrix[idx[-1], idx[0]] + np.sum(distance_matrix[idx[:-1], idx[1:]])

    def two_opt_steepest(tour):
        improved = True
        while improved:
            improved = False
            best_gain = 0.0
            best_i = best_j = None
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
                tour[i+1:j+1] = reversed(tour[i+1:j+1])
                improved = True
        return tour

    def double_bridge(tour):
        cuts = sorted(np.random.choice(range(1, n), 3, replace=False))
        seg0 = tour[:cuts[0]]
        seg1 = tour[cuts[0]:cuts[1]]
        seg2 = tour[cuts[1]:cuts[2]]
        seg3 = tour[cuts[2]:]
        return seg0 + seg2 + seg1 + seg3

    def nearest_insertion():
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            best_city = None
            best_cost = float('inf')
            best_pos = 0
            for city in unvisited:
                min_inc = float('inf')
                pos_best = 0
                for pos in range(len(tour) + 1):
                    if pos == 0:
                        inc = distance_matrix[tour[-1], city] + distance_matrix[city, tour[0]] - distance_matrix[tour[-1], tour[0]]
                    elif pos == len(tour):
                        inc = distance_matrix[tour[-1], city] + distance_matrix[city, tour[0]] - distance_matrix[tour[-1], tour[0]]
                    else:
                        inc = distance_matrix[tour[pos-1], city] + distance_matrix[city, tour[pos]] - distance_matrix[tour[pos-1], tour[pos]]
                    if inc < min_inc:
                        min_inc = inc
                        pos_best = pos
                if min_inc < best_cost:
                    best_cost = min_inc
                    best_city = city
                    best_pos = pos_best
            tour.insert(best_pos, best_city)
            unvisited.remove(best_city)
        return tour

    best_tour = None
    best_dist = float('inf')
    num_restarts = 10
    max_cycles = 10
    stall_limit = 2

    for _ in range(num_restarts):
        tour = nearest_insertion()
        cur_dist = total_dist(tour)
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_tour = np.array(tour)
            report_best_tour(best_tour)

        no_improve = 0
        for cycle in range(max_cycles):
            tour = two_opt_steepest(tour)
            cur_dist = total_dist(tour)
            if cur_dist < best_dist - 1e-12:
                best_dist = cur_dist
                best_tour = np.array(tour)
                report_best_tour(best_tour)
                no_improve = 0
            else:
                no_improve += 1

            if cycle < max_cycles - 1:
                if no_improve >= stall_limit:
                    tour = double_bridge(tour)
                    no_improve = 0
                else:
                    seg_len = np.random.randint(2, n//4 + 2)
                    i = np.random.randint(0, n - seg_len)
                    tour[i:i+seg_len] = reversed(tour[i:i+seg_len])

    return best_tour